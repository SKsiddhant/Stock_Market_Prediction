import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("tensorflow")
pytest.importorskip("prophet")
pytest.importorskip("statsmodels")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app5.py")


def fake_download(ticker, **kw):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2023-01-02", periods=260)
    close = 100 + np.cumsum(rng.normal(0.1, 1, len(idx)))
    df = pd.DataFrame({"Close": close, "Open": close}, index=idx)
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df


def test_app_trains_and_shows_holdout_scores(monkeypatch, tmp_path):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    assert not any(h.value.startswith("Held-out") for h in at.markdown)  # nothing trains until the button
    at.slider[0].set_value(1)
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("Held-out accuracy" in m.value for m in at.markdown)
    assert any("Naive" in str(df.value.index.tolist()) for df in at.dataframe)
    assert list(tmp_path.iterdir()) == []  # the app no longer litters the working dir with CSVs
