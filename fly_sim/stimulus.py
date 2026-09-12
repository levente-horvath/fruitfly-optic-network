"""Turn images into luminance movies on the eye's hexagonal column lattice."""

import gzip
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from fly_sim.connectome import DATA_DIR

DATASETS_DIR = DATA_DIR / "datasets"
CHECKERS_POSITIONS = DATASETS_DIR / "checkers" / "positions.parquet"


def column_positions(hex1: np.ndarray, hex2: np.ndarray) -> np.ndarray:
    """Planar (x, y) of each column, in units of the column spacing, centered on the eye.

    A column's six nearest neighbours are the offsets (±1, 0), (0, ±1) and ±(1, 1): 15% of
    synapses between column-assigned neurons cross to one of them, versus 0.5% to (1, -1) or
    (-1, 1). So the hex1 and hex2 axes are 120° apart.
    """
    xy = np.column_stack([hex1 - hex2 / 2, hex2 * np.sqrt(3) / 2])
    return xy - (xy.min(axis=0) + xy.max(axis=0)) / 2


@dataclass(frozen=True)
class Presentation:
    dt: float = 0.01  # seconds per frame
    blank: float = 0.1  # seconds of background before the image appears
    duration: float = 0.3  # seconds the image is on
    size: float = 22.0  # image width on the eye, in column spacings (the eye is ~28 wide, ~33 tall)
    background: float = 0.0  # MNIST-style datasets have black backgrounds
    jitter: float = 1.0  # flash: largest shift per axis, in column spacings
    jitter_interval: float = 0.05  # flash: seconds between shifts
    speed: float = 40.0  # moving: column spacings per second (~5° per column)


def flash_jitter(image: np.ndarray, xy: np.ndarray, p: Presentation, rng: np.random.Generator) -> np.ndarray:
    """Blank, then the image centered on the eye, jumping to a new small random offset at intervals.

    Returns luminance per frame and column, shape (frames, columns).
    """
    blank_frames, on_frames = round(p.blank / p.dt), round(p.duration / p.dt)
    hold = max(1, round(p.jitter_interval / p.dt))
    offsets = rng.uniform(-p.jitter, p.jitter, size=(-(-on_frames // hold), 2))
    offsets[0] = 0.0  # the flash itself lands centered
    offsets = np.repeat(offsets, hold, axis=0)[:on_frames]
    return _movie(image, xy, offsets, blank_frames, p)


def moving(image: np.ndarray, xy: np.ndarray, p: Presentation, rng: np.random.Generator) -> np.ndarray:
    """Blank, then the image drifting at constant speed in a random direction, bouncing inside the eye."""
    blank_frames, on_frames = round(p.blank / p.dt), round(p.duration / p.dt)
    limit = (xy.max(axis=0) - xy.min(axis=0)) / 2 - p.size / 2
    position = rng.uniform(-limit, limit)
    angle = rng.uniform(0, 2 * np.pi)
    velocity = p.speed * p.dt * np.array([np.cos(angle), np.sin(angle)])

    offsets = np.empty((on_frames, 2))
    for frame in range(on_frames):
        offsets[frame] = position
        position = position + velocity
        bounced = np.abs(position) > limit
        velocity[bounced] *= -1
        position = np.clip(position, -limit, limit)
    return _movie(image, xy, offsets, blank_frames, p)


def _movie(image, xy, offsets, blank_frames, p):
    height, width = image.shape
    scale = width / p.size  # image pixels per column spacing
    # Each column averages light over roughly one column spacing (FWHM), so blur before sampling.
    blurred = gaussian_filter(image.astype(np.float32), sigma=scale / 2.355, mode="constant", cval=p.background)

    frames = np.full((blank_frames + len(offsets), len(xy)), p.background, dtype=np.float32)
    for t, (dx, dy) in enumerate(offsets, start=blank_frames):
        cols = (xy[:, 0] - dx) * scale + (width - 1) / 2
        rows = -(xy[:, 1] - dy) * scale + (height - 1) / 2  # image rows run downward
        frames[t] = map_coordinates(blurred, [rows, cols], order=1, mode="constant", cval=p.background)
    return frames


def load_dataset(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Images scaled to [0, 1] and integer labels from an MNIST-format dataset in data/datasets/."""
    prefix = {"train": "train", "test": "t10k"}[split]
    images = _read_idx(DATASETS_DIR / name / f"{prefix}-images-idx3-ubyte.gz")
    labels = _read_idx(DATASETS_DIR / name / f"{prefix}-labels-idx1-ubyte.gz")
    return images.astype(np.float32) / 255, labels.astype(np.int64)


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)



# Checkers pieces are told apart by two cues that survive the eye's ~1-column blur: brightness for
# the side, and disc size for man versus king. A ring or a crown would be finer than one column.
PIECE = {  # board value -> (radius in squares, luminance)
    1: (0.34, 0.95),  # the side to move's man
    2: (0.46, 0.95),  # its king
    -1: (0.34, 0.45),  # the opponent's man
    -2: (0.46, 0.45),  # the opponent's king
}
DARK_SQUARE = 0.12  # the playable squares, so the grid is visible even where the board is empty

# A second encoding, chosen by experiments/encoding_search.py. Four luminance steps instead of two,
# a wider size contrast, and -- the change that mattered most by far -- twice as long to look at it:
# 93.0% of boards are then read with no square wrong, against 65.9% for PIECE at 0.3 s.
PIECE_V2 = {1: (0.30, 1.00), 2: (0.50, 0.75), -1: (0.30, 0.42), -2: (0.50, 0.20)}
BOARD_PIXELS = (16, 12)  # per square (rows, columns): the eye is ~1.15x taller than wide, and
# stretching the board to match leaves every playable square at least 7 columns of the eye.


def render_board(board: np.ndarray, pixels: tuple[int, int] = BOARD_PIXELS, piece: dict | None = None,
                 dark_square: float = DARK_SQUARE) -> np.ndarray:
    """Draw an 8x8 checkers board (values as in fly_sim.checkers.to_array) as a luminance image."""
    piece = piece or PIECE
    height, width = pixels
    rows, cols = np.mgrid[0 : 8 * height, 0 : 8 * width] / np.array([[[height]], [[width]]])
    image = np.where((rows.astype(int) + cols.astype(int)) % 2 == 1, dark_square, 0.0).astype(np.float32)

    for (row, col), value in np.ndenumerate(board):
        if value:
            radius, luminance = piece[int(value)]
            image[np.hypot(rows - (row + 0.5), cols - (col + 0.5)) <= radius] = luminance
    return image
