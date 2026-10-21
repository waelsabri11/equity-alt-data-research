"""Data-quality checks and coverage report for the price panel.

Each check returns counts; `clean_prices` applies the fixes and returns a mask of
usable (member and price-valid) stock-days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def recycled_tickers(adj_close: pd.DataFrame, intervals: pd.DataFrame, tol_days: int = 30) -> list[str]:
    """Tickers whose Yahoo history starts after their last index-membership interval ended.

    Yahoo reuses symbols: e.g. a symbol freed by an acquired company and later given to a
    new listing. Such histories belong to a different company and are dropped.
    """
    first = adj_close.apply(lambda s: s.first_valid_index())
    out = []
    for t, g in intervals.groupby("ticker"):
        if t not in first.index or first[t] is None or pd.isna(first[t]):
            continue
        last_end = g["end"].max() if g["end"].notna().all() else pd.NaT
        if pd.notna(last_end) and first[t] > last_end - pd.Timedelta(days=tol_days):
            out.append(t)
    return out


def stale_mask(close: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """True on stock-days inside runs of >= min_run identical closes.

    Yahoo pads some pre-listing histories (e.g. AMCR, SW) with a constant price.
    """
    same = close.diff().eq(0) & close.notna()
    out = pd.DataFrame(False, index=close.index, columns=close.columns)
    for c in same.columns:
        s = same[c].astype(int)
        run = s.groupby((s != s.shift()).cumsum()).transform("sum") * s
        hit = run >= min_run - 1
        out[c] = hit | hit.shift(-1, fill_value=False)  # include the first day of the run
    return out


def unexplained_jumps(close: pd.DataFrame, split: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Daily moves beyond +/-thr (log) in the split-adjusted close."""
    lr = np.log(close).diff()
    flags = lr.abs() > np.log(1 + thr)
    return flags & (split.reindex_like(flags).fillna(1.0) == 1.0)


def clean_prices(px: dict[str, pd.DataFrame], member: pd.DataFrame, intervals: pd.DataFrame
                 ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    close, adj = px["close"].copy(), px["adj_close"].copy()
    rep: dict = {}
    rep["tickers_in_universe"] = int(member.columns.size)
    has = adj.reindex(columns=member.columns).notna().any()
    rep["tickers_with_yahoo_data"] = int(has.sum())
    rec = recycled_tickers(adj, intervals)
    rep["recycled_tickers_dropped"] = rec
    adj[rec] = np.nan
    close[rec] = np.nan
    bad = (close <= 0) | (adj <= 0)
    rep["non_positive_prices"] = int(bad.sum().sum())
    close, adj = close.mask(bad), adj.mask(bad)
    stale = stale_mask(close)
    rep["stale_stock_days_masked(>=5 equal closes)"] = int(stale.sum().sum())
    close, adj = close.mask(stale), adj.mask(stale)
    jumps = unexplained_jumps(close, px["split"])
    rep["daily_moves_over_50pct_without_split"] = int(jumps.sum().sum())
    ret = adj.pct_change(fill_method=None)
    rep["daily_returns_over_100pct_or_below_-75pct"] = int(((ret > 1) | (ret < -0.75)).sum().sum())

    m = member.reindex(index=adj.index).fillna(False).astype(bool)
    valid = adj.reindex(columns=m.columns).notna()
    usable = m & valid
    md = m.sum().sum()
    rep["member_days"] = int(md)
    rep["member_days_with_price_pct"] = round(100 * usable.sum().sum() / md, 2)
    yearly = pd.DataFrame({"member_days": m.groupby(m.index.year).sum().sum(axis=1),
                           "with_price": usable.groupby(usable.index.year).sum().sum(axis=1)})
    yearly["coverage_pct"] = (100 * yearly["with_price"] / yearly["member_days"]).round(1)
    rep["coverage_by_year"] = yearly
    # Survivorship view: of names that left the index, how many still have prices?
    left = intervals.groupby("ticker")["end"].apply(lambda e: e.notna().all())
    left_names = left[left].index
    rep["names_removed_from_index"] = int(len(left_names))
    rep["removed_names_with_data_pct"] = round(100 * valid[left_names.intersection(valid.columns)].any().mean(), 1)
    cur_names = left[~left].index
    rep["current_names_with_data_pct"] = round(100 * valid[cur_names.intersection(valid.columns)].any().mean(), 1)
    out = dict(px)
    out["close"], out["adj_close"] = close, adj
    return out, usable, rep
