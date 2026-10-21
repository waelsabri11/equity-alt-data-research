"""Cointegrated-pairs statistical arbitrage, walk-forward.

Each window: a formation period (252 days) and the following trading period (126 days).
Formation (past data only):
  * candidates: same Fama-French-12 industry, index members with full price history;
  * prefilter the 20 lowest sum-of-squared-distance pairs per industry (normalised prices);
  * Engle-Granger test on log prices; keep p < 0.05, best 20 by p-value;
  * hedge ratio beta and spread mean / std frozen from formation.
Trading: z = (log A - beta log B - mu) / sigma. Enter at |z| > 2, exit at |z| < 0.5,
stop at |z| > 4, close at period end. A signal seen at the close of day d is executed at
the close of d+1. Each pair gets 1/20 of capital, legs are dollar weighted 1/(1+|beta|)
and |beta|/(1+|beta|); costs are charged per leg on each entry and exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .config import (PAIRS_COST_BPS, PAIRS_FORMATION, PAIRS_MAX, PAIRS_PREFILTER, PAIRS_PVALUE,
                     PAIRS_TRADING, Z_ENTRY, Z_EXIT, Z_STOP)


def select_pairs(logp: pd.DataFrame, sector: pd.Series) -> list[dict]:
    norm = logp - logp.iloc[0]
    out = []
    for s, names in sector.reindex(logp.columns).groupby(sector.reindex(logp.columns)).groups.items():
        if s == "Unknown" or len(names) < 2:
            continue
        X = norm[list(names)].values
        sq = (X ** 2).sum(axis=0)
        ssd = sq[:, None] + sq[None, :] - 2 * X.T @ X
        iu = np.triu_indices(len(names), 1)
        order = np.argsort(ssd[iu])[:PAIRS_PREFILTER]
        for k in order:
            a, b = names[iu[0][k]], names[iu[1][k]]
            ya, yb = logp[a].values, logp[b].values
            p = coint(ya, yb, trend="c", maxlag=1, autolag=None)[1]
            if p < PAIRS_PVALUE:
                beta, c = np.polyfit(yb, ya, 1)
                spread = ya - beta * yb
                out.append({"a": a, "b": b, "sector": s, "p": p, "beta": beta,
                            "mu": spread.mean(), "sd": spread.std(ddof=1)})
    out.sort(key=lambda d: d["p"])
    chosen, used = [], set()
    for d in out:  # each stock in at most one pair
        if d["a"] in used or d["b"] in used or d["beta"] <= 0:
            continue
        chosen.append(d)
        used |= {d["a"], d["b"]}
        if len(chosen) == PAIRS_MAX:
            break
    return chosen


def trade_pair(pa: pd.Series, pb: pd.Series, pr: dict, cost_bps: float = PAIRS_COST_BPS):
    """Daily returns (gross, net) of one pair on its capital slot, plus trade log."""
    z = ((np.log(pa) - pr["beta"] * np.log(pb)) - pr["mu"]) / pr["sd"]
    ra, rb = pa.pct_change().fillna(0).values, pb.pct_change().fillna(0).values
    wa, wb = 1 / (1 + pr["beta"]), pr["beta"] / (1 + pr["beta"])
    n = len(z)
    pos = np.zeros(n)  # position held over day i: decided at close i-2, traded at close i-1
    state = 0
    for i in range(n - 2):
        zi = z.iloc[i]
        if state == 0 and np.isfinite(zi) and Z_ENTRY < abs(zi) < Z_STOP:
            state = -np.sign(zi)  # z high: spread rich -> short A, long B
        elif state != 0 and (not np.isfinite(zi) or abs(zi) < Z_EXIT or abs(zi) > Z_STOP):
            state = 0
        pos[i + 2] = state
    gross = pos * (wa * ra - wb * rb)
    # position changes (including the forced close at the end of the period), both legs
    change = np.abs(np.diff(np.r_[0, pos, 0]))
    cost = change[:-1] * (wa + wb) * cost_bps / 1e4
    cost[-1] += change[-1] * (wa + wb) * cost_bps / 1e4
    trades = int(((np.r_[0, pos[:-1]] == 0) & (pos != 0)).sum())
    return gross, gross - cost, trades, pos


def walk_forward(adj_close: pd.DataFrame, member: pd.DataFrame, sector: pd.Series, start: str) -> dict:
    idx = adj_close.index
    i0 = idx.searchsorted(pd.Timestamp(start))
    gross_all, net_all, log, windows = [], [], [], []
    for f0 in range(i0 - PAIRS_FORMATION, len(idx) - PAIRS_FORMATION - 5, PAIRS_TRADING):
        form = idx[f0:f0 + PAIRS_FORMATION]
        trade = idx[f0 + PAIRS_FORMATION:f0 + PAIRS_FORMATION + PAIRS_TRADING]
        if len(trade) < 5:
            break
        names = member.columns[member.loc[form[-1]].values]
        px = adj_close.loc[form, names]
        px = px.loc[:, px.notna().all()]
        chosen = select_pairs(np.log(px), sector)
        tp = adj_close.loc[form[-1:].append(trade)]
        g_win = pd.DataFrame(0.0, index=trade, columns=range(PAIRS_MAX))
        n_win = g_win.copy()
        n_oos_coint = 0
        for k, pr in enumerate(chosen):
            pa, pb = tp[pr["a"]].ffill(), tp[pr["b"]].ffill()
            g, n, ntr, pos = trade_pair(pa, pb, pr)
            g_win[k], n_win[k] = g[1:], n[1:]
            ya, yb = np.log(pa.iloc[1:]), np.log(pb.iloc[1:])
            if ya.notna().all() and yb.notna().all():
                n_oos_coint += coint(ya.values, yb.values, trend="c", maxlag=1, autolag=None)[1] < PAIRS_PVALUE
            held = pos[1:] != 0
            log.append({"window_start": trade[0], "a": pr["a"], "b": pr["b"], "sector": pr["sector"],
                        "p_formation": pr["p"], "trades": ntr, "days_in_position": int(held.sum()),
                        "pnl_gross": float(np.prod(1 + g[1:]) - 1), "pnl_net": float(np.prod(1 + n[1:]) - 1)})
        windows.append({"trade_start": trade[0], "trade_end": trade[-1], "pairs": len(chosen),
                        "still_cointegrated_oos": n_oos_coint})
        # Capital split equally over PAIRS_MAX slots; unused slots sit in cash (0 return).
        gross_all.append(g_win.mean(axis=1))
        net_all.append(n_win.mean(axis=1))
    return {"gross": pd.concat(gross_all), "net": pd.concat(net_all),
            "pairs": pd.DataFrame(log), "windows": pd.DataFrame(windows)}
