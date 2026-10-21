"""Earnings-surprise signal from SEC filings.

SUE (standardized unexpected earnings, seasonal random walk):
    SUE_q = (EPS_q - EPS_{q-4}) / std(EPS_k - EPS_{k-4}, previous 8 quarters)

Timing is the point. The same EPS number appears twice on EDGAR:
  * in the earnings press release (8-K, Item 2.02, Exhibit 99.1), and
  * weeks later in the 10-Q / 10-K XBRL data.
The value comes from XBRL (first-filed figure, i.e. the number originally reported),
but the signal is dated at the 8-K release, as if the figure had been read from the
press release on the day it came out. That is exactly what the
llama3-finreport-extraction model does (diluted EPS F1 0.939 on 300 held-out releases);
`llm_crosscheck` measures how often its extracted EPS agrees with the XBRL figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fundamentals import quarterly


def eps_quarterly(facts: pd.DataFrame) -> pd.DataFrame:
    dil = quarterly(facts, "eps_diluted")
    bas = quarterly(facts, "eps_basic")
    key = ["ticker", "end"]
    bas = bas[~bas.set_index(key).index.isin(dil.set_index(key).index)]
    return pd.concat([dil, bas], ignore_index=True).sort_values(key).reset_index(drop=True)


def release_dates(eps: pd.DataFrame, filings: pd.DataFrame, max_lag_days: int = 90) -> pd.DataFrame:
    """Attach the earnings-release date: first 8-K with Item 2.02 filed after the period
    end and no later than the 10-Q/10-K. Falls back to the 10-Q/10-K filing date."""
    rel = filings[filings["items"].fillna("").str.contains("2.02", regex=False)]
    rel = rel[["ticker", "filingDate", "accessionNumber"]].sort_values("filingDate")
    by_t = {t: g for t, g in rel.groupby("ticker")}
    ann, src, accn = [], [], []
    for r in eps.itertuples(index=False):
        g = by_t.get(r.ticker)
        hit = None
        if g is not None:
            w = g[(g["filingDate"] > r.end) & (g["filingDate"] <= r.filed)
                  & (g["filingDate"] <= r.end + pd.Timedelta(days=max_lag_days))]
            if len(w):
                hit = w.iloc[0]
        if hit is not None:
            ann.append(hit["filingDate"]); src.append("8-K"); accn.append(hit["accessionNumber"])
        else:
            ann.append(r.filed); src.append("10-Q/K"); accn.append(None)
    out = eps.copy()
    out["release"] = pd.to_datetime(ann)
    out["release_src"] = src
    out["release_accn"] = accn
    return out


def compute_sue(eps: pd.DataFrame, min_hist: int = 4, window: int = 8) -> pd.DataFrame:
    """SUE per (ticker, quarter). The scale uses only seasonal differences of quarters
    before the current one."""
    out = []
    for _, g in eps.groupby("ticker"):
        g = g.sort_values("end").reset_index(drop=True)
        ends = g["end"].values
        vals = g["val"].values
        diffs = np.full(len(g), np.nan)
        for i in range(len(g)):
            lag = (ends[i] - ends[:i]) / np.timedelta64(1, "D")
            j = np.where((lag >= 350) & (lag <= 380))[0]
            if len(j):
                diffs[i] = vals[i] - vals[j[-1]]
        s = np.full(len(g), np.nan)
        for i in range(len(g)):
            past = diffs[max(0, i - window):i]
            past = past[np.isfinite(past)]
            if np.isfinite(diffs[i]) and len(past) >= min_hist:
                s[i] = diffs[i] / max(past.std(ddof=1), 0.01)
        g["sue"] = s
        out.append(g.dropna(subset=["sue"]))
    return pd.concat(out, ignore_index=True)


def announcement_returns(events: pd.DataFrame, adj_close: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Market-adjusted return from the close before the release day to the close of the
    trading day after it. Usable from the close of that day (`ear_avail`)."""
    idx = adj_close.index
    out = events.copy()
    pos = idx.searchsorted(out["release"].values, side="left")  # release day (or next trading day)
    ear = np.full(len(out), np.nan)
    avail = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    lr = np.log(adj_close)
    lm = np.log(market)
    cols = {c: k for k, c in enumerate(adj_close.columns)}
    arr = lr.values
    marr = lm.values
    for k, (t, p) in enumerate(zip(out["ticker"].values, pos)):
        if t not in cols or p - 1 < 0 or p + 1 >= len(idx):
            continue
        c = cols[t]
        r = arr[p + 1, c] - arr[p - 1, c]
        m = marr[p + 1] - marr[p - 1]
        if np.isfinite(r) and np.isfinite(m):
            ear[k] = r - m
            avail[k] = idx[p + 1]
    out["ear"] = ear
    out["ear_avail"] = avail
    return out


def llm_crosscheck(eps_events: pd.DataFrame, finreport_root: Path) -> dict | None:
    """Compare EPS extracted by the QLoRA model from 8-K press releases with the
    first-filed XBRL EPS used here. Matching is on (cik, release accession)."""
    pred = finreport_root / "runs/qlora/predictions_test.jsonl"
    meta = finreport_root / "data/processed/test.jsonl"
    if not (pred.exists() and meta.exists()):
        return None
    m = {}
    for line in meta.open():
        d = json.loads(line)
        m[d["id"]] = d
    rows = []
    for line in pred.open():
        p = json.loads(line)
        d = m.get(p["id"])
        if d is None:
            continue
        try:
            eps = (json.loads(p["raw"]).get("diluted_eps") or {}).get("value")
        except (json.JSONDecodeError, AttributeError):
            eps = None
        rows.append((d["ticker"].upper().replace(".", "-"), p["id"].split("-", 1)[1], eps))
    llm = pd.DataFrame(rows, columns=["ticker", "accn", "llm_eps"])
    ev = eps_events.dropna(subset=["release_accn"])[["ticker", "release_accn", "val"]]
    j = llm.merge(ev, left_on=["ticker", "accn"], right_on=["ticker", "release_accn"])
    j = j.dropna(subset=["llm_eps"])
    if j.empty:
        return {"matched_releases": 0}
    agree = (j["llm_eps"] - j["val"]).abs() <= 0.005 + 1e-9
    return {"llm_test_releases": int(len(llm)), "matched_releases_in_universe": int(len(j)),
            "eps_agree_within_half_cent_pct": round(100 * agree.mean(), 1)}


def event_study(ev: pd.DataFrame, adj_close: pd.DataFrame, market: pd.Series, member: pd.DataFrame,
                start: str, horizon: int = 60) -> tuple[pd.DataFrame, dict]:
    """Market-adjusted log returns by SUE quintile around each earnings release.

    reaction      close(d-1) -> close(d+1), d = 8-K release day
    drift_8k      close(d+1) -> close(d+1+h): tradable once the release is read
    drift_10q     close(f+1) -> close(f+1+h): tradable once the 10-Q/10-K XBRL is filed
    Quintiles are formed within each calendar quarter of release (no pooling across time).
    Only events by index members, released from `start` on, with an 8-K date.
    """
    idx = adj_close.index
    lp = np.log(adj_close).sub(np.log(market), axis=0)  # log price relative to SPY
    arr, cols = lp.values, {c: k for k, c in enumerate(lp.columns)}
    e = ev[(ev["release_src"] == "8-K") & (ev["release"] >= start)].dropna(subset=["sue"]).copy()
    pd_ = idx.searchsorted(e["release"].values, side="left")
    pf = idx.searchsorted(e["filed"].values, side="left")

    def window(p0, p1):
        out = np.full(len(e), np.nan)
        for k, (t, a, b) in enumerate(zip(e["ticker"].values, p0, p1)):
            c = cols.get(t)
            if c is None or a < 0 or b >= len(idx):
                continue
            out[k] = arr[b, c] - arr[a, c]
        return out

    e["reaction"] = window(pd_ - 1, pd_ + 1)
    e["drift_8k"] = window(pd_ + 1, pd_ + 1 + horizon)
    e["drift_10q"] = window(pf + 1, pf + 1 + horizon)
    e["8k_to_10q"] = window(pd_ + 1, pf + 1)
    rel_day = idx[np.clip(pd_, 0, len(idx) - 1)]
    mem = [bool(member.at[d, t]) if t in member.columns else False for d, t in zip(rel_day, e["ticker"])]
    e = e[np.array(mem)].dropna(subset=["reaction", "drift_8k", "drift_10q"])
    e["q"] = e.groupby(e["release"].dt.to_period("Q"))["sue"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    cols_ = ["reaction", "8k_to_10q", "drift_8k", "drift_10q"]
    tab = 100 * e.groupby("q")[cols_].mean()
    tab.index = [f"Q{int(i)}" for i in tab.index]
    per_q = e.groupby([e["release"].dt.to_period("Q"), "q"])[cols_].mean().unstack("q")
    spread = {}
    for c in cols_:
        s = per_q[c][5] - per_q[c][1]
        spread[c] = {"Q5-Q1_pct": 100 * s.mean(), "t_stat": s.mean() / s.std() * np.sqrt(len(s))}
    tab.loc["Q5-Q1"] = [spread[c]["Q5-Q1_pct"] for c in cols_]
    tab.loc["t-stat (across quarters)"] = [spread[c]["t_stat"] for c in cols_]
    stats = {"events": int(len(e)), "quarters": int(per_q.shape[0]),
             "median_trading_days_8k_to_10q": float(np.median(pf - pd_)) if len(e) else np.nan}
    return tab, stats


def pead_weights(ev: pd.DataFrame, index: pd.DatetimeIndex, member: pd.DataFrame, start: str,
                 timing: str = "release", hold: int = 60, lookback_days: int = 365) -> dict:
    """Event-driven post-earnings-drift book.

    An event becomes tradable at the close of the trading day after its date (`release`
    for the 8-K, `filed` for the 10-Q/10-K). It goes long if its SUE is at or above the
    80th percentile of SUE over events dated in the previous `lookback_days` (strictly
    before), short if at or below the 20th; held `hold` trading days. Each day the long
    and short sides are equal-weighted at +1 / -1.
    """
    e = ev[(ev["release_src"] == "8-K")].dropna(subset=["sue"]).sort_values(timing).reset_index(drop=True)
    d = e[timing].values
    s = e["sue"].values
    side = np.zeros(len(e))
    lo_ptr = 0
    for k in range(len(e)):
        while d[lo_ptr] < d[k] - np.timedelta64(lookback_days, "D"):
            lo_ptr += 1
        hi = np.searchsorted(d, d[k], side="left")   # strictly earlier dates only
        past = s[lo_ptr:hi]
        if len(past) < 200:
            continue
        p20, p80 = np.quantile(past, [0.2, 0.8])
        side[k] = 1 if s[k] >= p80 else (-1 if s[k] <= p20 else 0)
    e["side"] = side
    e = e[e["side"] != 0]
    entry = index.searchsorted(e[timing].values, side="right")  # first trading day after the date
    e = e.assign(entry=entry, exit=entry + hold)
    e = e[(e["entry"] < len(index))]
    ok = [bool(member.at[index[i], t]) if t in member.columns else False for i, t in zip(e["entry"], e["ticker"])]
    e = e[np.array(ok)]
    start_i = index.searchsorted(pd.Timestamp(start))
    weights = {}
    ent, ext, tick, sd = e["entry"].values, e["exit"].values, e["ticker"].values, e["side"].values
    for i in range(start_i, len(index)):
        live = (ent <= i) & (ext > i)
        longs = np.unique(tick[live & (sd > 0)])
        shorts = np.unique(tick[live & (sd < 0)])
        shorts = np.setdiff1d(shorts, longs)
        if len(longs) == 0 or len(shorts) == 0:
            weights[index[i]] = pd.Series(dtype=float)
            continue
        weights[index[i]] = pd.concat([pd.Series(1 / len(longs), index=longs),
                                       pd.Series(-1 / len(shorts), index=shorts)])
    return weights
