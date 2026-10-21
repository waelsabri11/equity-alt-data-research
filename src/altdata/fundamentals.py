"""Point-in-time fundamentals from SEC XBRL company facts.

Every reported value carries the date it was filed. A value is usable at date t only
if it was filed strictly before t, and for each fiscal period the first filed value
is kept (later restatements are ignored, as they were unknown at the time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INTERIM
from .edgar import Edgar

CONCEPTS = {
    "eps_diluted": ("us-gaap", "EarningsPerShareDiluted", "USD/shares"),
    "eps_basic": ("us-gaap", "EarningsPerShareBasic", "USD/shares"),
    "net_income": ("us-gaap", "NetIncomeLoss", "USD"),
    "equity": ("us-gaap", "StockholdersEquity", "USD"),
    "assets": ("us-gaap", "Assets", "USD"),
    "shares_diluted": ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
}
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"}

# Fama-French 12 industries from SIC codes (Ken French, Siccodes12).
_FF12 = [
    ("NoDur", [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ("Durbl", [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
               (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),
    ("Manuf", [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
               (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
               (3830, 3839), (3860, 3899)]),
    ("Enrgy", [(1200, 1399), (2900, 2999)]),
    ("Chems", [(2800, 2829), (2840, 2899)]),
    ("BusEq", [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ("Telcm", [(4800, 4899)]),
    ("Utils", [(4900, 4949)]),
    ("Shops", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Hlth", [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ("Money", [(6000, 6999)]),
]


def ff12(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    for name, ranges in _FF12:
        if any(lo <= s <= hi for lo, hi in ranges):
            return name
    return "Other"


def tidy_facts(facts: dict) -> pd.DataFrame:
    rows = []
    for key, (ns, tag, unit) in CONCEPTS.items():
        for f in facts.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {}).get(unit, []):
            if f.get("form") not in FORMS:
                continue
            rows.append((key, f.get("start"), f["end"], float(f["val"]), f["filed"], f["form"]))
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "val", "filed", "form"])
    for c in ["start", "end", "filed"]:
        df[c] = pd.to_datetime(df[c])
    return df


def onboard(tickers: list[str], ticker_cik: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download (or read cached) company facts and submissions. Returns
    facts (long), 8-K filings with items, and a ticker info table (cik, sic, ff12)."""
    fpath, apath, ipath = INTERIM / "facts.parquet", INTERIM / "filings_8k.parquet", INTERIM / "info.parquet"
    if fpath.exists():
        return pd.read_parquet(fpath), pd.read_parquet(apath), pd.read_parquet(ipath)
    ed = Edgar()
    facts, filings, info = [], [], []
    for i, t in enumerate(tickers):
        cik = ticker_cik.get(t)
        if cik is None:
            info.append((t, None, None, "Unknown"))
            continue
        sub = ed.submissions(cik)
        sic = sub.get("sic") if sub else None
        info.append((t, cik, sic, ff12(sic)))
        if sub:
            r = sub["filings"]["recent"]
            df = pd.DataFrame({k: r[k] for k in ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]})
            df = df[df["form"].isin(["8-K", "8-K/A"])]
            df["ticker"] = t
            filings.append(df)
        cf = ed.company_facts(cik)
        if cf:
            df = tidy_facts(cf)
            df["ticker"] = t
            facts.append(df)
        if i % 50 == 0:
            print(f"  edgar {i}/{len(tickers)} ({ed.n_requests} network requests)", flush=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    facts = pd.concat(facts, ignore_index=True)
    filings = pd.concat(filings, ignore_index=True)
    filings["filingDate"] = pd.to_datetime(filings["filingDate"])
    info = pd.DataFrame(info, columns=["ticker", "cik", "sic", "ff12"])
    info["sic"] = info["sic"].astype("string")
    facts.to_parquet(fpath, index=False)
    filings.to_parquet(apath, index=False)
    info.to_parquet(ipath, index=False)
    return facts, filings, info


def _first_filed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values("filed").drop_duplicates(keys, keep="first")


def quarterly(facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    """First-filed quarterly values per (ticker, period end); fiscal Q4 is derived as
    annual minus the three reported quarters and becomes available with the 10-K."""
    d = facts[facts["concept"] == concept].dropna(subset=["start"]).copy()
    d["days"] = (d["end"] - d["start"]).dt.days
    q = _first_filed(d[d["days"].between(75, 105)], ["ticker", "end"])
    a = _first_filed(d[d["days"].between(340, 390)], ["ticker", "end"])
    out = [q[["ticker", "start", "end", "val", "filed"]]]
    derived = []
    for t, ga in a.groupby("ticker"):
        gq = q[q["ticker"] == t]
        have = set(gq["end"])
        for _, r in ga.iterrows():
            if any(abs((r["end"] - e).days) <= 7 for e in have):
                continue
            inside = gq[(gq["start"] >= r["start"] - pd.Timedelta(days=7)) & (gq["end"] < r["end"] - pd.Timedelta(days=60))]
            if len(inside) != 3:
                continue
            derived.append((t, inside["end"].max() + pd.Timedelta(days=1), r["end"],
                            r["val"] - inside["val"].sum(), max(r["filed"], inside["filed"].max())))
    if derived:
        out.append(pd.DataFrame(derived, columns=["ticker", "start", "end", "val", "filed"]))
    res = pd.concat(out, ignore_index=True).sort_values(["ticker", "end"]).reset_index(drop=True)
    return res


def instant(facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    d = facts[facts["concept"] == concept]
    return _first_filed(d, ["ticker", "end"])[["ticker", "end", "val", "filed"]].sort_values(["ticker", "end"])


def ttm(q: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month sums from four consecutive quarters; available when the
    last of the four was filed."""
    rows = []
    for t, g in q.groupby("ticker"):
        g = g.sort_values("end").reset_index(drop=True)
        for i in range(3, len(g)):
            w = g.iloc[i - 3:i + 1]
            if (w["end"].iloc[-1] - w["end"].iloc[0]).days > 300:  # ~273 days for 4 consecutive quarters
                continue
            rows.append((t, w["end"].iloc[-1], w["val"].sum(), w["filed"].max()))
    return pd.DataFrame(rows, columns=["ticker", "end", "val", "filed"])


def asof_panel(events: pd.DataFrame, dates: pd.DatetimeIndex, tickers, avail_col: str = "filed",
               val_col: str = "val", max_age_days: int | None = 400) -> pd.DataFrame:
    """(date x ticker) panel of the latest value whose availability date is strictly
    before each date. Among available values, the one with the latest period end wins."""
    out = pd.DataFrame(np.nan, index=dates, columns=pd.Index(tickers))
    for t, g in events.groupby("ticker"):
        if t not in out.columns:
            continue
        g = g.sort_values(avail_col)
        # latest period end known so far (handles late-filed older periods)
        g = g[g["end"] >= g["end"].cummax()] if "end" in g else g
        av = g[avail_col].values
        pos = np.searchsorted(av, dates.values, side="left") - 1  # strictly before
        ok = pos >= 0
        vals = np.full(len(dates), np.nan)
        vals[ok] = g[val_col].values[pos[ok]]
        if max_age_days is not None:
            age = np.full(len(dates), np.inf)
            age[ok] = (dates.values[ok] - av[pos[ok]]) / np.timedelta64(1, "D")
            vals[age > max_age_days] = np.nan
        out[t] = vals
    return out
