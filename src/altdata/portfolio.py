"""Portfolio construction, daily backtest with costs, performance and factor exposures."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import ASSUMED_IC, BORROW_BPS_PER_YEAR, COST_BPS, COV_WINDOW, MV_RISK_AVERSION, NAME_CAP


# ---------------------------------------------------------------- backtest
def backtest(weights: dict[pd.Timestamp, pd.Series], adj_close: pd.DataFrame,
             cost_bps: float = COST_BPS, borrow_bps: float = BORROW_BPS_PER_YEAR) -> pd.DataFrame:
    """Weights set at the close of each rebalance date and held (drifting) until the next.

    Daily P&L of a held book: sum_i w_i * (P_i(d) / P_i(t) - 1). A stock that stops
    trading is carried at its last price. Costs: cost_bps x one-way traded notional
    against the drifted book; borrow: borrow_bps per year on short notional.
    """
    px = adj_close.ffill()
    dates = sorted(weights)
    end_date = px.index[-1]
    out = []
    prev_drift = pd.Series(dtype=float)
    nav = 1.0
    for k, t in enumerate(dates):
        w = weights[t].dropna()
        w = w[w != 0]
        t_next = dates[k + 1] if k + 1 < len(dates) else end_date
        days = px.index[(px.index > t) & (px.index <= t_next)]
        if len(days) == 0:
            continue
        trade = w.sub(prev_drift.reindex(w.index.union(prev_drift.index)).fillna(0), fill_value=0).abs().sum()
        cost = cost_bps / 1e4 * trade
        p0 = px.loc[t, w.index]
        w = w[p0.notna()]
        rel = px.loc[days, w.index] / p0[w.index]
        pnl = (rel - 1).mul(w, axis=1).sum(axis=1)               # cumulative since t, per unit NAV
        short = w[w < 0].abs().sum()
        borrow = borrow_bps / 1e4 / 252 * short * np.arange(1, len(days) + 1)
        gross_path = 1 + pnl
        net_path = (1 - cost) * (1 + pnl) - borrow
        prev_g = np.r_[1.0, gross_path.values[:-1]]
        prev_n = np.r_[1.0, net_path.values[:-1]]
        out.append(pd.DataFrame({"gross": gross_path.values / prev_g - 1,
                                 "net": net_path.values / prev_n - 1,
                                 "turnover": np.r_[trade, np.zeros(len(days) - 1)],
                                 "gross_exposure": w.abs().sum()}, index=days))
        prev_drift = w * rel.iloc[-1] / (1 + pnl.iloc[-1])
    return pd.concat(out)


def perf(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = r.dropna()
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    out = {"ann_return_pct": 100 * ann, "ann_vol_pct": 100 * vol, "sharpe": ann / vol if vol else np.nan,
           "max_drawdown_pct": 100 * dd.min(), "years": len(r) / 252}
    if turnover is not None:
        out["turnover_ann"] = turnover.sum() / (len(r) / 252)
    return out


# ---------------------------------------------------------------- building blocks
def sleeve_weights(z: pd.Series) -> pd.Series:
    """Long-short weights proportional to a (sector-demeaned) score, gross 2."""
    z = z.dropna()
    g = z.abs().sum()
    return z / g * 2 if g > 0 else z


def ledoit_wolf(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    lw = LedoitWolf().fit(returns.fillna(0.0).values)
    return lw.covariance_, lw.shrinkage_


def erc_weights(cov: np.ndarray) -> np.ndarray:
    """Equal-risk-contribution (risk parity) weights summing to 1, via the convex
    problem min 0.5 y'Cy - mean(log y) (Spinu 2013), then w = y / sum(y)."""
    n = cov.shape[0]
    f = lambda y: 0.5 * y @ cov @ y - np.log(y).mean()
    g = lambda y: cov @ y - 1 / (n * y)
    res = minimize(f, np.full(n, 1 / np.sqrt(np.trace(cov))), jac=g, method="L-BFGS-B",
                   bounds=[(1e-10, None)] * n, options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 1000})
    return res.x / res.x.sum()


def betas(returns: pd.DataFrame, market: pd.Series) -> pd.Series:
    m = market - market.mean()
    r = returns.sub(returns.mean())
    return (r.mul(m, axis=0)).sum() / (m ** 2).sum()


def mean_variance(score: pd.Series, returns: pd.DataFrame, market: pd.Series, sector: pd.Series,
                  w_prev: pd.Series | None, cost_bps: float = COST_BPS,
                  risk_aversion: float = MV_RISK_AVERSION, cap: float = NAME_CAP) -> tuple[pd.Series, dict]:
    """max  alpha'w - lambda w'Sigma w - cost |w - w_prev|
       s.t. sum w = 0, beta'w = 0, sector sums = 0, |w_i| <= cap, sum|w| <= 2.

    alpha_i = IC * sigma_i(monthly) * z_i (Grinold); Sigma = Ledoit-Wolf on daily returns,
    scaled to monthly. The shrunk covariance (1-d) S + d mu I is used in factored form.
    """
    names = score.dropna().index.intersection(returns.columns)
    R = returns[names]
    R = R.loc[:, R.notna().mean() > 0.8]
    names = R.columns
    X = R.fillna(0.0).values
    X = X - X.mean(axis=0)
    T, n = X.shape
    lw = LedoitWolf(assume_centered=True).fit(X)
    d = lw.shrinkage_
    S_diag = (X ** 2).mean(axis=0)
    mu = S_diag.mean()
    sig_m = np.sqrt(((1 - d) * S_diag + d * mu) * 21)
    z = score[names].values
    z = (z - z.mean()) / z.std()
    alpha = ASSUMED_IC * sig_m * z
    b = betas(R.fillna(0.0), market.reindex(R.index).fillna(0.0)).values
    sec = sector.reindex(names).fillna("Unknown").values
    w0 = (w_prev.reindex(names).fillna(0.0).values if w_prev is not None else np.zeros(n))

    w = cp.Variable(n)
    risk = 21 * ((1 - d) * cp.sum_squares(X @ w) / T + d * mu * cp.sum_squares(w))
    obj = alpha @ w - risk_aversion * risk - cost_bps / 1e4 * cp.norm1(w - w0)
    cons = [cp.sum(w) == 0, b @ w == 0, cp.abs(w) <= cap, cp.norm1(w) <= 2]
    for s in np.unique(sec):
        cons.append(cp.sum(w[np.where(sec == s)[0]]) == 0)
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    wv = pd.Series(np.where(np.abs(w.value) < 1e-6, 0.0, w.value), index=names)
    return wv, {"shrinkage": d, "ex_ante_vol_ann": float(np.sqrt(12 * risk.value)), "status": prob.status}


# ---------------------------------------------------------------- exposures
def factor_regression(r: pd.Series, factors: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    import statsmodels.api as sm

    j = pd.concat([r.rename("r"), factors], axis=1, join="inner").dropna()
    X = sm.add_constant(j[factors.columns])
    fit = sm.OLS(j["r"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = pd.DataFrame({"coef": fit.params, "t": fit.tvalues})
    out.loc["const", "coef"] *= 252  # annualised alpha
    out = out.rename(index={"const": "alpha_ann"})
    out.loc["R2", "coef"] = fit.rsquared
    return out
