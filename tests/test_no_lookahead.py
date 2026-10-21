"""Every signal / position at date t must be unchanged when data from t onwards changes."""
import numpy as np
import pandas as pd

from altdata import factors, pairs
from altdata.fundamentals import asof_panel


def _perturb_from(df, t, seed=1):
    out = df.copy()
    rng = np.random.default_rng(seed)
    m = out.index >= t
    out.loc[m] = out.loc[m] * rng.uniform(0.5, 2.0, out.loc[m].shape)
    return out


def test_price_signals_ignore_prices_from_rebalance_day(prices):
    dates = factors.rebalance_dates(prices.index, "2017-01-01")
    t = dates[10]
    base = factors.price_signals(prices, dates)
    pert = factors.price_signals(_perturb_from(prices, t), dates)
    for k in base:
        pd.testing.assert_series_equal(base[k].loc[t], pert[k].loc[t])
        # and the perturbation does reach later dates (the test is not vacuous)
    assert not base["REV"].loc[dates[11]].equals(pert["REV"].loc[dates[11]])


def test_asof_panel_is_strictly_before():
    ev = pd.DataFrame({"ticker": ["A", "A"], "end": pd.to_datetime(["2020-03-31", "2020-06-30"]),
                       "filed": pd.to_datetime(["2020-05-01", "2020-08-03"]), "val": [1.0, 2.0]})
    d = pd.to_datetime(["2020-05-01", "2020-05-04", "2020-08-03", "2020-08-04"])
    p = asof_panel(ev, d, ["A"])["A"].tolist()
    assert np.isnan(p[0]) and p[1] == 1.0 and p[2] == 1.0 and p[3] == 2.0


def test_asof_ignores_late_filed_older_period():
    # an amendment for an OLD period filed later must not replace a newer period
    ev = pd.DataFrame({"ticker": ["A"] * 3, "end": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31"]),
                       "filed": pd.to_datetime(["2020-05-01", "2020-08-03", "2020-09-01"]), "val": [1.0, 2.0, 9.0]})
    p = asof_panel(ev, pd.to_datetime(["2020-10-01"]), ["A"])["A"].iloc[0]
    assert p == 2.0


def test_event_signal_not_usable_on_release_day():
    ev = pd.DataFrame({"ticker": ["A"], "end": pd.to_datetime(["2020-03-31"]),
                       "release": pd.to_datetime(["2020-04-30"]), "sue": [1.5]})
    d = pd.to_datetime(["2020-04-30", "2020-05-01"])
    s = factors.event_signal(ev, "sue", "release", d, ["A"])["A"]
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 1.5


def test_fundamental_signal_ignores_filings_from_rebalance_day(prices):
    dates = factors.rebalance_dates(prices.index, "2017-01-01")
    t = dates[5]
    tick = prices.columns[:3]
    rows = []
    for k, tk in enumerate(tick):
        for q, end in enumerate(pd.date_range("2015-03-31", "2018-09-30", freq="QE")):
            filed = end + pd.Timedelta(days=35)
            for c, v in [("net_income", 10 + q), ("shares_diluted", 1e6), ("assets", 1e4), ("equity", 5e3)]:
                rows.append((c, end - pd.Timedelta(days=90) if c in ("net_income", "shares_diluted") else pd.NaT,
                             end, float(v * (k + 1)), filed, "10-Q", tk))
    facts = pd.DataFrame(rows, columns=["concept", "start", "end", "val", "filed", "form", "ticker"])
    split = pd.DataFrame(1.0, index=prices.index, columns=tick)
    base = factors.fundamental_signals(facts, prices[tick], split, dates)
    later = (facts["filed"] >= t) & (facts["concept"] == "net_income")
    f2 = facts.copy()
    f2.loc[later, "val"] *= 7
    pert = factors.fundamental_signals(f2, _perturb_from(prices[tick], t), split, dates)
    for k in ["BM", "EP", "ROA"]:
        pd.testing.assert_series_equal(base[k].loc[t], pert[k].loc[t])
    assert not np.allclose(base["ROA"].loc[dates[-1]], pert["ROA"].loc[dates[-1]])


def test_pair_positions_use_only_past_prices():
    rng = np.random.default_rng(3)
    n = 130
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    a = b * np.exp(rng.normal(0, 0.03, n))
    pr = {"beta": 1.0, "mu": 0.0, "sd": 0.03}
    _, _, _, pos = pairs.trade_pair(a, b, pr)
    k = 60
    a2 = a.copy()
    a2.iloc[k:] *= 1.5
    _, _, _, pos2 = pairs.trade_pair(a2, b, pr)
    # position on day i is decided with prices up to day i-2
    np.testing.assert_array_equal(pos[:k + 2], pos2[:k + 2])
