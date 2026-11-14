import numpy as np
import pytest


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(816)
    c = rng.normal(size=(72, 3))
    x = 2 * c[:, 0] - c[:, 1] + rng.normal(size=72)
    y = 0.4 * x[:, None] + c @ rng.normal(size=(3, 5)) + rng.normal(size=(72, 5))
    return x, y, c


@pytest.fixture
def cv_data():
    rng = np.random.default_rng(912)
    c = rng.normal(size=(64, 3))
    x = c @ rng.normal(size=(3, 12)) + rng.normal(size=(64, 12))
    y = 2 * c[:, 0] + 0.8 * x[:, 0] + rng.normal(size=64)
    return x, y, c, np.arange(64) % 4

