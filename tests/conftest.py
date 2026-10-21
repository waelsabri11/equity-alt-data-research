import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", "2018-12-31")
    cols = [f"S{i}" for i in range(40)]
    r = rng.normal(0.0003, 0.015, (len(idx), len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
