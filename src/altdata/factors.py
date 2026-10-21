"""Monthly cross-sectional signals.

Information set at rebalance date t (last trading day of a month):
  prices through the close of t-1 (the previous trading day),
  filings with a filing date strictly before t.
Positions are traded at the close of t and earn returns from t onwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import asof_panel, instant, quarterly, ttm
from .prices import split_factor_after


def rebalance_dates(index: pd.DatetimeIndex, start: str) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    me = pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)
    return me[me >= pd.Timestamp(start)]


def _prev_day(index: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index[index.searchsorted(dates, side="left") - 1]


def price_signals(adj_close: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    idx = adj_close.index
    prev = _prev_day(idx, dates)
    pos = idx.get_indexer(prev)
    a = adj_close.values

    def at(k):
        p = pos - k
        out = np.full((len(dates), a.shape[1]), np.nan)
        ok = p >= 0
        out[ok] = a[p[ok]]
        return pd.DataFrame(out, index=dates, columns=adj_close.columns)

    p0, p21, p252 = at(0), at(21), at(252)
    ret = adj_close.pct_change(fill_method=None)
    vol = ret.rolling(252, min_periods=200).std().reindex(prev)
    vol.index = dates
    return {
        "MOM": p21 / p252 - 1,          # 12-1 momentum
        "REV": -(p0 / p21 - 1),         # 1-month reversal
        "LOWVOL": -vol,
    }


def fundamental_signals(facts: pd.DataFrame, close: pd.DataFrame, split: pd.DataFrame,
                        dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    tickers = close.columns
    idx = close.index
    prev = _prev_day(idx, dates)
    px = close.reindex(prev)
    px.index = dates

    # Market cap = Yahoo split-adjusted close x reported share count x splits after the
    # filing date. Yahoo applies every later split retroactively to past prices; the
    # split factor undoes exactly that, giving the market cap as it stood on t-1.
    sh = quarterly(facts, "shares_diluted")
    fa = split_factor_after(split)
    adj = []
    for r in sh.itertuples(index=False):
        if r.ticker in fa.columns:
            k = fa.index.searchsorted(r.filed, side="right") - 1
            adj.append(r.val * (fa[r.ticker].iloc[k] if k >= 0 else fa[r.ticker].iloc[0]))
        else:
            adj.append(np.nan)
    sh = sh.assign(val=adj)
    shares = asof_panel(sh, dates, tickers)
    mcap = px * shares

    ni_ttm = asof_panel(ttm(quarterly(facts, "net_income")), dates, tickers)
    equity = asof_panel(instant(facts, "equity"), dates, tickers)
    assets = asof_panel(instant(facts, "assets"), dates, tickers)
    bm = (equity / mcap).where(equity > 0)
    ep = ni_ttm / mcap
    roa = ni_ttm / assets
    return {"BM": bm, "EP": ep, "ROA": roa, "MCAP": mcap}


def event_signal(events: pd.DataFrame, value: str, avail: str, dates, tickers, max_age_days: int = 95
                 ) -> pd.DataFrame:
    ev = events.dropna(subset=[value, avail]).rename(columns={avail: "_avail", value: "_v"})
    return asof_panel(ev[["ticker", "end", "_avail", "_v"]], dates, tickers, avail_col="_avail",
                      val_col="_v", max_age_days=max_age_days)


def neutralize(sig: pd.DataFrame, sector: pd.Series, universe: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """Per date: keep universe names, rank-normalise to z, demean within sector."""
    s = sig.where(universe)
    r = s.rank(axis=1, pct=True)
    z = (r - r.mean(axis=1).values[:, None]) / r.std(axis=1).values[:, None]
    z = z.clip(-winsor, winsor)
    sec = sector.reindex(z.columns).fillna("Unknown")
    return z.T.groupby(sec.values).transform(lambda x: x - x.mean()).T


def composite(zs: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    stack = np.stack([zs[n].values for n in names])
    with np.errstate(all="ignore"):
        m = np.nanmean(stack, axis=0)
    return pd.DataFrame(m, index=zs[names[0]].index, columns=zs[names[0]].columns)
