"""Play checkers against the real optic lobe, in a browser.

    uv run python experiments/play_server.py     # then open http://localhost:8770

Every move the fly makes is a real simulation: each legal afterstate is drawn as a board, shown to
the 49,401-neuron right eye, and scored by the ridge value head from experiments/checkers_play.py.
The page shows the network's own neurons lighting up with the rates it just computed.
"""

import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkers_play import board_seed
from fly_sim import checkers as ck
from fly_sim import probe, search
from fly_sim.connectome import ANNOTATIONS, CACHE_DIR
from fly_sim.stimulus import PIECE_V2, render_board
from fly_sim.vision import BOARD_V2, Eye, frame_window

PAGE = Path(__file__).resolve().parents[1] / "viz" / "play.html"
VOXEL_UM = 0.008  # MaleCNS coordinates are 8 nm voxels
HOST = os.environ.get("FLY_SIM_HOST", "127.0.0.1")  # behind a reverse proxy in production
PORT = int(os.environ.get("FLY_SIM_PORT", "8770"))
# One board at a time. A move keeps per-move state (the neuron rates it will display), and each
# simulation takes a core and ~0.5 GB; two visitors moving at once would overwrite each other's
# state and starve the machine. A second request waits briefly, then is told to try again.
SIMULATION = threading.Lock()
WAIT_SECONDS = 25


def whole_cns(lobe) -> tuple[np.ndarray, np.ndarray, int]:
    """Soma coordinates of every traced neuron that has one, and where each sits in the simulation.

    The simulated network is one optic lobe out of a whole central nervous system, so the rest of
    the brain and nerve cord is drawn too, unlit -- that contrast is the point. Only the neuron
    table is needed here, not the 1 GB weight matrix.
    """
    neurons = pd.read_parquet(CACHE_DIR / "neurons.parquet", columns=["bodyId"])
    somas = feather.read_table(ANNOTATIONS, columns=["bodyId", "somaLocation", "tosomaLocation"]).to_pandas()
    merged = neurons.merge(somas, on="bodyId", how="left")
    location = merged["somaLocation"].where(merged["somaLocation"].notna(), merged["tosomaLocation"])
    shown = location.notna().to_numpy()

    xyz = np.stack(location[shown].to_numpy()).astype(np.float64) * VOXEL_UM
    x, y, z = (xyz - (xyz.min(0) + xyz.max(0)) / 2).T
    positions = np.column_stack([x, -z, y]).astype(np.float32)  # body axis vertical: brain on top

    simulated = np.full(len(neurons), -1, dtype=np.int32)
    simulated[lobe.neurons["cns_index"].to_numpy()] = np.arange(len(lobe.neurons), dtype=np.int32)
    return positions, simulated[shown], len(neurons)


class Fly:
    """The opponent, as strong as this project has made it.

    The eye looks at each board for 0.6 s under the v2 encoding; a decoder trained on dense
    per-square targets reads the board back off the visual projection neurons; a linear evaluator
    trained on boards corrupted the way that decoder corrupts them judges it; and forced captures
    are played out before anything is judged. Measured over 40 games each: 97.5% against a random
    player, 53.8% against a depth-1 alpha-beta player with a hand-tuned evaluation, 26.2% against
    depth 2 -- where the version this replaced scored 61.3%, 1.2% and 1.2%.
    """

    def __init__(self):
        self.eye = Eye("fly")
        d = np.load(CACHE_DIR / "strong_decoder.npz")
        self.decoder = probe.Ridge(d["mean"], d["std"], d["coef"], d["target_mean"], 0.0, 0.0)
        self.weights = np.load(CACHE_DIR / "strong_evaluator.npz")["weights"]
        self.eye.model  # build the rate model now rather than on the first move
        self.window = frame_window(BOARD_V2)
        self.positions, self.simulated, self.traced = whole_cns(self.eye.lobe)
        self.is_vpn = np.zeros(len(self.eye.lobe.neurons), dtype=bool)
        self.is_vpn[self.eye.lobe.outputs] = True
        self.rest = self.eye.model.baseline(self.eye.params)[0]
        self.rates: dict[ck.Position, np.ndarray] = {}
        print(f"{len(self.positions):,} of {self.traced:,} traced neurons have a soma; "
              f"{(self.simulated >= 0).sum():,} of them are in the simulated right eye", flush=True)

    def look(self, positions: list[ck.Position]) -> np.ndarray:
        """Show each board to the eye; keep every neuron's rate, return the visual projection features."""
        images = [render_board(ck.to_array(p), piece=PIECE_V2) for p in positions]
        movies = self.eye.movies(images, [board_seed(p) for p in positions], BOARD_V2)
        rates, _ = self.eye.model.run(movies, BOARD_V2.dt, self.window, self.eye.params)
        self.rates.update(zip(positions, rates.astype(np.float16)))
        return (rates[:, self.eye.lobe.outputs] - self.eye.resting).astype(np.float64)

    def evaluate(self, positions: list[ck.Position]) -> np.ndarray:
        """Value to the side to move, read through the eye: decode the board, round it, judge it."""
        board = self.decoder.predict(self.look(positions)).reshape(len(positions), 32, 5).argmax(2)
        # the evaluator scores a board for whoever just moved, so negate for the side to move
        return -(np.eye(5)[board].reshape(len(positions), -1) @ self.weights)

    def choose(self, position: ck.Position) -> dict:
        started = time.perf_counter()
        before = len(self.rates)
        self.rates.clear()
        moves, scores = search.move_scores(position, self.evaluate, depth=0)
        best = int(np.argmax(scores))
        shown = moves[best].result
        if shown not in self.rates:  # a forced capture meant the eye judged a later board
            self.look([shown])
        return {
            "move": list(moves[best].path),
            "candidates": sorted(
                [{"path": list(m.path), "score": float(v)} for m, v in zip(moves, scores)],
                key=lambda c: -c["score"],
            ),
            "rates": b64(self.rates[shown]),
            "result": shown,
            "seconds": time.perf_counter() - started,
            "boards": len(self.rates) or before,
        }


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def state(position: ck.Position, whose_turn: str, note: str = "") -> dict:
    moves = ck.legal_moves(position)
    return {
        "position": list(position),
        "board": ck.to_array(position).tolist(),
        "legal": [list(m.path) for m in moves],
        "turn": whose_turn,
        "over": not moves,
        "note": note,
    }


class Handler(BaseHTTPRequestHandler):
    fly: Fly = None

    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/index.html"):
            return self.send_error(404)
        payload = {
            "positions": b64(self.fly.positions),
            "simulated": b64(self.fly.simulated),
            "isVpn": b64(self.fly.is_vpn.astype(np.uint8)),
            "rest": b64(self.fly.rest.astype(np.float16)),
            "neurons": int(len(self.fly.positions)),
            "simCount": int((self.fly.simulated >= 0).sum()),
            "traced": int(self.fly.traced),
            "start": state(ck.START, "you"),
        }
        body = PAGE.read_text().replace("__PAYLOAD__", json.dumps(payload)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/move":
            return self.send_error(404)
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        position = ck.Position(*request["position"])

        if request.get("path"):  # the human moves first, if the request carries one
            wanted = tuple(request["path"])
            move = next((m for m in ck.legal_moves(position) if m.path == wanted), None)
            if move is None:
                return self.reply({"error": "illegal move"})
            position = move.result

        if not ck.legal_moves(position):
            return self.reply({**state(position, "you"), "outcome": "you win"})

        if not SIMULATION.acquire(timeout=WAIT_SECONDS):
            return self.reply({"error": "the fly is thinking about another game -- try again in a moment"})
        try:
            thought = self.fly.choose(position)
        finally:
            SIMULATION.release()
        after = thought["result"]
        reply = {
            **state(after, "you"),
            "flyMove": thought["move"],
            "candidates": thought["candidates"],
            "rates": thought["rates"],
            "seconds": thought["seconds"],
            "boards": thought["boards"],
        }
        if reply["over"]:
            reply["outcome"] = "the fly wins"
        self.reply(reply)

    def reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    Handler.fly = Fly()
    print(f"serving http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
