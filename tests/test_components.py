import numpy as np
import pandas as pd

from altdata import earnings, evaluation, factors, portfolio, universe
from altdata.fundamentals import quarterly, ttm


def test_universe_reverse_walk():
    changes = pd.DataFrame({"date": pd.to_datetime(["2020-01-10", "2018-06-01"]),
                            "added": ["C", "B"], "removed": ["B", "X"]})
    iv, issues = universe.membership_intervals(["A", "C"], changes, "2021-01-01", "2017-01-01")
    m = universe.member_mask(iv, pd.to_datetime(["2017-06-01", "2019-01-01", "2020-06-01"]), ["A", "B", "C", "X"])
    assert m.loc["2017-06-01"].tolist() == [True, False, False, True]
    assert m.loc["2019-01-01"].tolist() == [True, True, False, False]
    assert m.loc["2020-06-01"].tolist() == [True, False, True, False]
    assert issues == {"added_not_member": 0, "removed_already_member": 0}


def _facts(vals_q, annual, concept="eps_diluted", restated=None):
    rows = []
    for (s, e, v, f) in vals_q:
        rows.append((concept, pd.Timestamp(s), pd.Timestamp(e), v, pd.Timestamp(f), "10-Q", "A"))
    s, e, v, f = annual
    rows.append((concept, pd.Timestamp(s), pd.Timestamp(e), v, pd.Timestamp(f), "10-K", "A"))
    if restated:
        rows.append((concept, pd.Timestamp(restated[0]), pd.Timestamp(restated[1]), restated[2],
                     pd.Timestamp(restated[3]), "10-Q", "A"))
    return pd.DataFrame(rows, columns=["concept", "start", "end", "val", "filed", "form", "ticker"])


def test_quarterly_first_filed_and_q4_derivation():
    q = [("2019-01-01", "2019-03-31", 1.0, "2019-05-01"), ("2019-04-01", "2019-06-30", 1.0, "2019-08-01"),
         ("2019-07-01", "2019-09-30", 1.0, "2019-11-01")]
    f = _facts(q, ("2019-01-01", "2019-12-31", 5.0, "2020-02-20"),
               restated=("2019-01-01", "2019-03-31", 9.0, "2020-05-01"))
    out = quarterly(f, "eps_diluted")
    assert out.loc[out["end"] == "2019-03-31", "val"].item() == 1.0       # original, not restated
    q4 = out[out["end"] == "2019-12-31"]
    assert q4["val"].item() == 2.0 and q4["filed"].item() == pd.Timestamp("2020-02-20")
    assert ttm(out)["val"].item() == 5.0


def test_sue_scale_uses_past_quarters_only():
    ends = pd.date_range("2015-03-31", periods=16, freq="QE")
    rng = np.random.default_rng(0)
    eps = pd.DataFrame({"ticker": "A", "end": ends, "val": rng.normal(1, 0.2, 16),
                        "filed": ends + pd.Timedelta(days=40)})
    base = earnings.compute_sue(eps).set_index("end")["sue"]
    e2 = eps.copy()
    e2.loc[e2.index[-1], "val"] += 10
    pert = earnings.compute_sue(e2).set_index("end")["sue"]
    pd.testing.assert_series_equal(base.iloc[:-1], pert.iloc[:-1])
    assert pert.iloc[-1] > base.iloc[-1]


def test_neutralize_demeans_within_sector():
    d = pd.to_datetime(["2020-01-31"])
    sig = pd.DataFrame([np.arange(10.0)], index=d, columns=list("abcdefghij"))
    sector = pd.Series(["x"] * 5 + ["y"] * 5, index=sig.columns)
    z = factors.neutralize(sig, sector, sig.notna())
    assert abs(z.iloc[0, :5].sum()) < 1e-12 and abs(z.iloc[0, 5:].sum()) < 1e-12


def test_forward_returns_alignment(prices):
    dates = factors.rebalance_dates(prices.index, "2016-01-01")
    fwd = evaluation.forward_returns(prices, dates, 1)
    exp = prices.loc[dates[4], "S0"] / prices.loc[dates[3], "S0"] - 1
    assert np.isclose(fwd.loc[dates[3], "S0"], exp)
    assert fwd.loc[dates[-1]].isna().all()


def test_backtest_buy_and_hold_with_costs(prices):
    dates = factors.rebalance_dates(prices.index, "2016-01-01")[:3]
    w = {d: pd.Series({"S0": 1.0, "S1": -1.0}) for d in dates}
    bt = portfolio.backtest(w, prices, cost_bps=10, borrow_bps=0)
    t0, t1 = dates[0], dates[1]
    g = (1 + bt["gross"].loc[t0:t1].iloc[1:]).prod() if False else (1 + bt["gross"].loc[(bt.index > t0) & (bt.index <= t1)]).prod()
    exp = 1 + (prices.loc[t1, "S0"] / prices.loc[t0, "S0"] - 1) - (prices.loc[t1, "S1"] / prices.loc[t0, "S1"] - 1)
    assert np.isclose(g, exp)
    n = (1 + bt["net"].loc[(bt.index > t0) & (bt.index <= t1)]).prod()
    assert np.isclose(n, (1 - 0.002) * exp)       # entry cost: 2 units of notional x 10 bps


def test_erc_equalises_risk_contributions():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    cov = A @ A.T + 0.5 * np.eye(4)
    w = portfolio.erc_weights(cov)
    rc = w * (cov @ w)
    assert np.allclose(rc, rc.mean(), rtol=1e-3) and np.isclose(w.sum(), 1)


def test_mean_variance_constraints(prices):
    rets = prices.pct_change().iloc[-252:]
    mkt = rets.mean(axis=1)
    score = pd.Series(np.random.default_rng(2).normal(size=40), index=prices.columns)
    sector = pd.Series(["a", "b"] * 20, index=prices.columns)
    w, diag = portfolio.mean_variance(score, rets, mkt, sector, None, cap=0.1)
    beta = portfolio.betas(rets, mkt)
    assert diag["status"] == "optimal"
    assert abs(w.sum()) < 1e-6 and abs((w * beta[w.index]).sum()) < 1e-6
    assert all(abs(w.groupby(sector[w.index]).sum()) < 1e-6)
    assert w.abs().max() <= 0.1 + 1e-6 and w.abs().sum() <= 2 + 1e-6
    assert np.corrcoef(w, score[w.index])[0, 1] > 0.3
