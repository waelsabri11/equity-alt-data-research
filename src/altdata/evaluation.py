"""Signal evaluation: IC / rank IC, IC decay, turnover, quantile spreads."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(adj_close: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int = 1) -> pd.DataFrame:
    """Return from rebalance date i+h-1 to i+h (month h after formation). A stock that
    stops trading inside the month is credited its return to the last available price."""
    px = adj_close.ffill(limit=None).reindex(dates)
    last_valid = adj_close.apply(lambda s: s.last_valid_index())
    alive = pd.DataFrame({c: dates <= (last_valid[c] if pd.notna(last_valid[c]) else dates[0] - pd.Timedelta(days=1))
                          for c in adj_close.columns}, index=dates)
    start = px.shift(-(horizon - 1)).where(alive.shift(-(horizon - 1), fill_value=False))
    end = px.shift(-horizon)
    return end / start - 1


def ic_series(signal: pd.DataFrame, fwd: pd.DataFrame, rank: bool = True) -> pd.Series:
    s, f = signal.align(fwd, join="inner")
    ok = s.notna() & f.notna()
    s, f = s.where(ok), f.where(ok)
    if rank:
        s, f = s.rank(axis=1), f.rank(axis=1)
    sc = s.sub(s.mean(axis=1), axis=0)
    fc = f.sub(f.mean(axis=1), axis=0)
    ic = (sc * fc).sum(axis=1) / np.sqrt((sc ** 2).sum(axis=1) * (fc ** 2).sum(axis=1))
    ic[ok.sum(axis=1) < 30] = np.nan
    return ic.dropna()


def summarize_ic(ic: pd.Series) -> dict:
    n = len(ic)
    return {"mean": ic.mean(), "std": ic.std(), "ICIR_ann": ic.mean() / ic.std() * np.sqrt(12),
            "t_stat": ic.mean() / ic.std() * np.sqrt(n), "hit_rate": (ic > 0).mean(), "months": n}


def quantile_returns(signal: pd.DataFrame, fwd: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    rows = {}
    for d in signal.index.intersection(fwd.index):
        s, f = signal.loc[d], fwd.loc[d]
        ok = s.notna() & f.notna()
        if ok.sum() < 50:
            continue
        b = pd.qcut(s[ok].rank(method="first"), q, labels=False) + 1
        rows[d] = f[ok].groupby(b).mean()
    return pd.DataFrame(rows).T


def top_quantile_turnover(signal: pd.DataFrame, q: int = 5) -> float:
    prev, tos = None, []
    for d in signal.index:
        s = signal.loc[d].dropna()
        if len(s) < 50:
            continue
        top = set(s[s >= s.quantile(1 - 1 / q)].index)
        if prev:
            tos.append(1 - len(top & prev) / len(top))
        prev = top
    return float(np.mean(tos)) if tos else np.nan


def rank_autocorr(signal: pd.DataFrame) -> float:
    r = signal.rank(axis=1)
    ac = [r.iloc[i].corr(r.iloc[i - 1]) for i in range(1, len(r))]
    return float(np.nanmean(ac))


def evaluate(signals: dict[str, pd.DataFrame], adj_close: pd.DataFrame, dates: pd.DatetimeIndex,
             decay_horizons=(1, 2, 3, 6, 9, 12)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fwds = {h: forward_returns(adj_close, dates, h) for h in decay_horizons}
    rows, decay, quant = [], {}, {}
    for name, sig in signals.items():
        ic = ic_series(sig, fwds[1], rank=True)
        pic = ic_series(sig, fwds[1], rank=False)
        qr = quantile_returns(sig, fwds[1])
        spread = qr[5] - qr[1]
        s = summarize_ic(ic)
        rows.append({"signal": name, "rank_IC": s["mean"], "IC_pearson": pic.mean(), "rank_IC_t": s["t_stat"],
                     "ICIR_ann": s["ICIR_ann"], "IC_hit": s["hit_rate"],
                     "Q5-Q1_ann_pct": 100 * 12 * spread.mean(),
                     "Q5-Q1_t": spread.mean() / spread.std() * np.sqrt(len(spread)),
                     "top_q_turnover": top_quantile_turnover(sig), "rank_autocorr": rank_autocorr(sig),
                     "avg_names": sig.notna().sum(axis=1).mean(), "months": s["months"]})
        decay[name] = {h: ic_series(sig, fwds[h]).mean() for h in decay_horizons}
        quant[name] = (100 * 12 * qr.mean()).round(2)
    return pd.DataFrame(rows).set_index("signal"), pd.DataFrame(decay).T, quant
