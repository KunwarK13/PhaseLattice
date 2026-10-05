from pathlib import Path

import pytest

from phaselattice import default_family, reconstruct
from phaselattice.synthetic import generate


@pytest.fixture(scope="session")
def example(tmp_path_factory):
    directory = tmp_path_factory.mktemp("observations")
    generate(directory, default_family(), 3, 410007, k_override=(-2, 0, 2), n_train=512, n_val=96)
    result = reconstruct(directory / "observations.json")
    return Path(directory), result
