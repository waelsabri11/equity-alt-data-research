"""Daily prices from Yahoo Finance (yfinance), cached as parquet.

Two price series are kept per ticker:
  close     split-adjusted, not dividend-adjusted (used with share counts)
  adj_close split- and dividend-adjusted (used for total returns)
plus volume and split ratios, so raw historical prices can be rebuilt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from .config import PRICE_END, PRICE_START, RAW


def download(tickers: list[str], start: str = PRICE_START, end: str = PRICE_END,
             chunk: int = 100) -> dict[str, pd.DataFrame]:
    path = RAW / "prices.parquet"
    if path.exists():
        long = pd.read_parquet(path)
    else:
        frames = []
        for i in range(0, len(tickers), chunk):
            batch = tickers[i:i + chunk]
            d = yf.download(batch, start=start, end=end, auto_adjust=False, actions=True,
                            progress=False, threads=True, group_by="column")
            if d.empty:
                continue
            d = d.stack(level=1, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
            frames.append(d)
        long = pd.concat(frames, ignore_index=True)
        long = long.rename(columns={"Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                                    "Stock Splits": "split", "Dividends": "dividend"})
        long = long[["date", "ticker", "close", "adj_close", "volume", "split", "dividend"]]
        long = long.dropna(subset=["close", "adj_close"], how="all")
        path.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(path, index=False)
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    out = {}
    for col in ["close", "adj_close", "volume", "split"]:
        out[col] = long.pivot(index="date", columns="ticker", values=col).sort_index()
    out["split"] = out["split"].fillna(0.0).replace(0.0, 1.0)
    return out


def split_factor_after(split: pd.DataFrame) -> pd.DataFrame:
    """Product of split ratios strictly after each date (raw price = close * this).

    Note this uses splits up to the end of the sample, so it is only for rebuilding
    historical raw prices, never as a signal.
    """
    rev = split.iloc[::-1].cumprod().iloc[::-1]
    return rev.shift(-1).fillna(1.0)


def splits_between(split: pd.DataFrame) -> pd.DataFrame:
    """Cumulative split product up to and including each date (for share-count adjustment)."""
    return split.cumprod()


def total_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    return adj_close.pct_change(fill_method=None)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def winsor(x: pd.Series, k: float = 3.0) -> pd.Series:
    return x.clip(-k, k)


def zscore(x: pd.Series) -> pd.Series:
    sd = x.std()
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * np.nan
