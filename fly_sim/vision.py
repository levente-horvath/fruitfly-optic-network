"""One stimulus-to-features protocol, shared by every experiment and every baseline network.

An `Eye` holds a network (the real optic lobe or a null model), its frozen rate dynamics and the
presentation protocol, and turns images into the two feature sets everything is compared on: the
visual projection neuron responses, and the raw luminance on the eye's columns.
"""

import numpy as np

from fly_sim.baselines import build
from fly_sim.optic_lobe import OpticLobe, right_eye
from fly_sim.rate_model import CALIBRATED, RateModel, RateParams
from fly_sim.stimulus import Presentation, column_positions, flash_jitter

# A checkers board fills the eye: 22 columns wide and, stretched to the eye's taller shape, 25 tall,
# which leaves every playable square at least 7 columns. The jitter is half a column, well under the
# ~3-column squares, so the readout cannot rely on the board landing in exactly one place.
BOARD = Presentation(size=22.0, jitter=0.5)
# The second protocol: the same board, looked at for twice as long. Perception is the ceiling on
# play, and this is what lifts whole-board accuracy from 66% to 93%.
BOARD_V2 = Presentation(size=22.0, jitter=0.5, duration=0.6)


class Eye:
    def __init__(self, network: str = "fly", seed: int = 0, params: RateParams = CALIBRATED, lobe: OpticLobe | None = None):
        lobe = lobe or right_eye()
        self.column_hex = lobe.neurons.loc[lobe.inputs, ["hex1", "hex2"]].drop_duplicates().to_numpy()
        self.xy = column_positions(self.column_hex[:, 0], self.column_hex[:, 1])
        self.network, self.params = network, params
        self.lobe = build(network, lobe, seed)
        self._model = None

    @property
    def model(self) -> RateModel:
        """Built on first use, so a pixels-only readout never pays for the dynamics."""
        if self._model is None:
            self._model = RateModel(self.lobe, self.column_hex)
        return self._model

    @property
    def resting(self) -> np.ndarray:
        """Output rates with no stimulus, which the features are measured against."""
        return self.model.baseline(self.params)[0][self.lobe.outputs]

    def movies(self, images, seeds, presentation: Presentation = BOARD, make_movie=flash_jitter) -> np.ndarray:
        return np.stack(
            [make_movie(image, self.xy, presentation, np.random.default_rng(seed)) for image, seed in zip(images, seeds)]
        )

    def columns(self, images, seeds, presentation: Presentation = BOARD, make_movie=flash_jitter) -> np.ndarray:
        """Mean luminance on each column while the image is up: the raw-pixel feature set."""
        start, stop = frame_window(presentation)
        return self.movies(images, seeds, presentation, make_movie)[:, start:stop].mean(axis=1)

    def look(self, images, seeds, presentation: Presentation = BOARD, make_movie=flash_jitter) -> tuple[np.ndarray, np.ndarray]:
        """Response to each image: visual projection neuron rates above rest, and mean column luminance."""
        movies = self.movies(images, seeds, presentation, make_movie)
        window = frame_window(presentation)
        mean, _ = self.model.run(movies, presentation.dt, window, self.params)
        return mean[:, self.lobe.outputs] - self.resting, movies[:, window[0] : window[1]].mean(axis=1)


def frame_window(presentation: Presentation) -> tuple[int, int]:
    """The frames the image is on screen, which is what the features average over."""
    return (
        round(presentation.blank / presentation.dt),
        round((presentation.blank + presentation.duration) / presentation.dt),
    )
