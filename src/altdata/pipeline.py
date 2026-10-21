"""End to end: onboarding -> signals -> evaluation -> pairs -> portfolios -> results/."""
from __future__ import annotations

import io
import json
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from . import earnings, evaluation, factors, fundamentals, pairs, portfolio, prices, quality, universe
from .config import (COST_BPS, EVAL_START, MARKET, RAW, RESULTS, SEED, SEED_CACHE, SEC_USER_AGENT)
from .edgar import Edgar

FF_URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}


def fama_french() -> pd.DataFrame:
    frames = []
    for k, url in FF_URLS.items():
        path = RAW / f"{k}.zip"
        if not path.exists():
            r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
        with zipfile.ZipFile(path) as z:
            txt = z.read(z.namelist()[0]).decode("latin-1")
        lines = txt.splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().startswith(",") or l.lower().startswith(",mkt") or ",Mom" in l or ",Mkt-RF" in l)
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), index_col=0)
        df = df[pd.to_numeric(df.index.astype(str).str.strip(), errors="coerce").notna()]
        df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
        df.columns = [c.strip() for c in df.columns]
        frames.append(df.astype(float) / 100)
    ff = pd.concat(frames, axis=1, join="inner")
    return ff.drop(columns=["RF"])


def _md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    cols = [d.index.name or ""] + [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in d.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)


def main() -> None:
    np.random.seed(SEED)
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    report: dict = {}

    # ------------------------------------------------------------ 1. universe + prices
    print("[1/7] universe and prices", flush=True)
    current, changes, snapshot = universe.fetch_wikipedia()
    iv, issues = universe.membership_intervals(current["ticker"].tolist(), changes, snapshot, "2013-01-01")
    tickers = sorted(iv["ticker"].unique())
    px = prices.download(tickers + [MARKET])
    member = universe.member_mask(iv, px["adj_close"].index, tickers)
    px, usable, q = quality.clean_prices(px, member, iv)
    q["change_table_inconsistencies"] = issues
    q["membership_counts"] = {str(d.date()): int(member.loc[d].sum()) for d in
                              member.index[member.index.searchsorted(pd.to_datetime(
                                  ["2014-01-02", "2018-01-02", "2022-01-03", "2026-06-30"]))]}
    adj, close = px["adj_close"], px["close"]
    mkt = adj[MARKET]

    # ------------------------------------------------------------ 2. EDGAR
    print("[2/7] SEC EDGAR facts and 8-K filings", flush=True)
    with_px = [t for t in tickers if t in adj.columns and adj[t].notna().any()]
    tmap = Edgar().tickers()
    for t, c in zip(current["ticker"], current["cik"]):
        tmap.setdefault(t, int(c))
    facts, filings, info = fundamentals.onboard(with_px, tmap)
    sector = info.set_index("ticker")["ff12"]
    eps = earnings.eps_quarterly(facts)
    ev = earnings.release_dates(eps, filings)
    ev = earnings.compute_sue(ev)
    ev = earnings.announcement_returns(ev, adj.drop(columns=[MARKET]), mkt)
    ev["filed_avail"] = ev["filed"]
    lag = (ev["filed"] - ev["release"]).dt.days
    in_window = ev[ev["end"] >= "2013-06-01"]
    q["edgar"] = {
        "tickers_with_price": len(with_px), "tickers_with_cik": int(info["cik"].notna().sum()),
        "tickers_with_xbrl_facts": int(facts["ticker"].nunique()),
        "quarterly_eps_events_since_2013": int(len(in_window)),
        "events_timed_by_8k_release_pct": round(100 * (in_window["release_src"] == "8-K").mean(), 1),
        "median_days_8k_to_10q": float(lag[ev["release_src"] == "8-K"].median()),
        "mean_days_8k_to_10q": round(float(lag[ev["release_src"] == "8-K"].mean()), 1),
    }
    es_tab, es_stats = earnings.event_study(ev, adj.drop(columns=[MARKET]), mkt, member, EVAL_START)
    es_tab.to_csv(RESULTS / "earnings_event_study.csv")
    q["event_study"] = es_stats
    llm = earnings.llm_crosscheck(ev, SEED_CACHE.parents[1])
    q["llm_crosscheck"] = llm

    # ------------------------------------------------------------ 3. signals
    print("[3/7] signals", flush=True)
    dates = factors.rebalance_dates(adj.index, EVAL_START)
    stocks = with_px
    prev = adj.index[adj.index.searchsorted(dates) - 1]
    univ = member.reindex(dates)[stocks] & adj[stocks].reindex(prev).set_axis(dates).notna()
    raw = factors.price_signals(adj[stocks], dates)
    raw.update(factors.fundamental_signals(facts, close[stocks], px["split"].reindex(columns=stocks).fillna(1.0), dates))
    raw["SUE"] = factors.event_signal(ev, "sue", "release", dates, stocks)
    raw["SUE_10Q_timing"] = factors.event_signal(ev, "sue", "filed_avail", dates, stocks)
    raw["EAR"] = factors.event_signal(ev, "ear", "ear_avail", dates, stocks)
    z = {k: factors.neutralize(v, sector, univ) for k, v in raw.items() if k != "MCAP"}
    z["VALUE"] = factors.composite(z, ["BM", "EP"])
    z["QUALITY"] = z["ROA"]
    z["EARNINGS"] = factors.composite(z, ["SUE", "EAR"])
    sleeves = ["MOM", "VALUE", "QUALITY", "EARNINGS"]
    z["COMPOSITE"] = factors.composite(z, sleeves)
    cov_rows = {k: round(100 * float((raw[k].where(univ).notna().sum().sum()) / univ.sum().sum()), 1)
                for k in ["MOM", "BM", "EP", "ROA", "SUE", "EAR"]}
    q["signal_coverage_pct_of_universe_months"] = cov_rows

    # ------------------------------------------------------------ 4. evaluation
    print("[4/7] factor evaluation", flush=True)
    order = ["MOM", "REV", "LOWVOL", "BM", "EP", "VALUE", "ROA", "SUE", "SUE_10Q_timing", "EAR",
             "EARNINGS", "COMPOSITE"]
    ic_tab, decay, quant = evaluation.evaluate({k: z[k] for k in order}, adj[stocks], dates)
    ic_tab.to_csv(RESULTS / "factor_ic.csv")
    decay.to_csv(RESULTS / "ic_decay.csv")
    pd.DataFrame(quant).T.to_csv(RESULTS / "quintile_returns_ann_pct.csv")
    # sub-periods
    sub = {}
    for name, (a, b) in {"2014-2019": ("2014", "2019"), "2020-2026": ("2020", "2026")}.items():
        fwd = evaluation.forward_returns(adj[stocks], dates)
        sub[name] = {k: evaluation.ic_series(z[k].loc[a:b], fwd.loc[a:b]).mean() for k in order}
    sub = pd.DataFrame(sub)
    sub.to_csv(RESULTS / "rank_ic_subperiods.csv")

    # ------------------------------------------------------------ 5. pairs
    print("[5/7] pairs walk-forward", flush=True)
    pr = pairs.walk_forward(adj[stocks], member[stocks].reindex(adj.index).fillna(False).astype(bool), sector, EVAL_START)
    pp = {"gross": portfolio.perf(pr["gross"]), "net": portfolio.perf(pr["net"])}
    pl = pr["pairs"]
    pairs_summary = {
        "windows": int(len(pr["windows"])), "pairs_traded": int(len(pl)),
        "avg_pairs_per_window": round(float(pr["windows"]["pairs"].mean()), 1),
        "pairs_still_cointegrated_in_trading_window_pct":
            round(100 * pr["windows"]["still_cointegrated_oos"].sum() / max(pr["windows"]["pairs"].sum(), 1), 1),
        "round_trips": int(pl["trades"].sum()),
        "pairs_with_a_trade_pct": round(100 * float((pl["trades"] > 0).mean()), 1),
        "win_rate_of_traded_pairs_net_pct": round(100 * float((pl.loc[pl.trades > 0, "pnl_net"] > 0).mean()), 1),
        "avg_capital_in_use_pct": round(100 * float((pl["days_in_position"].sum()) /
                                                    (len(pr["windows"]) * 20 * 126)), 1),
    }
    pl.to_csv(RESULTS / "pairs_trades.csv", index=False)
    pr["windows"].to_csv(RESULTS / "pairs_windows.csv", index=False)

    # ------------------------------------------------------------ 6. portfolios
    print("[6/7] portfolios", flush=True)
    rets = adj[stocks].pct_change(fill_method=None)
    mret = mkt.pct_change()
    sleeve_w = {s: {d: portfolio.sleeve_weights(z[s].loc[d]) for d in dates} for s in sleeves}
    bts = {s: portfolio.backtest(sleeve_w[s], adj[stocks]) for s in sleeves}
    sleeve_ret = pd.DataFrame({s: bts[s]["gross"] for s in sleeves})

    def combine(b, d):
        w = sum(b[k] * sleeve_w[s][d].reindex(stocks).fillna(0) for k, s in enumerate(sleeves))
        return portfolio.sleeve_weights(w[w != 0])

    ew_w = {d: combine(np.full(4, 0.25), d) for d in dates}
    erc_w, erc_b = {}, {}
    for d in dates:
        hist = sleeve_ret.loc[:d].iloc[:-1].tail(252)  # sleeve returns through t-1
        if len(hist) < 126:
            b = np.full(4, 0.25)
        else:
            cov, _ = portfolio.ledoit_wolf(hist)
            v = np.sqrt(np.diag(cov))
            b = portfolio.erc_weights(cov)
        erc_b[d] = b
        erc_w[d] = combine(b, d)
    mv_w, mv_diag, w_prev = {}, [], None
    for d in dates:
        p = adj.index[adj.index.searchsorted(d) - 1]
        win = rets.loc[:p].tail(252)
        score = z["COMPOSITE"].loc[d].dropna()
        w, diag = portfolio.mean_variance(score, win, mret.loc[win.index], sector, w_prev)
        mv_w[d], w_prev = w, w
        mv_diag.append({"date": d, **diag, "gross": float(w.abs().sum()), "names": int((w != 0).sum())})
    mv_diag = pd.DataFrame(mv_diag)
    bts["EW_sleeves"] = portfolio.backtest(ew_w, adj[stocks])
    bts["RiskParity_sleeves"] = portfolio.backtest(erc_w, adj[stocks])
    bts["MeanVariance_LW"] = portfolio.backtest(mv_w, adj[stocks])
    member_d = member[stocks].reindex(adj.index).fillna(False).astype(bool)
    pead_w = {tm: earnings.pead_weights(ev, adj.index, member_d, EVAL_START, timing=tm)
              for tm in ["release", "filed"]}
    bts["PEAD_8K_timing"] = portfolio.backtest(pead_w["release"], adj[stocks])
    bts["PEAD_10Q_timing"] = portfolio.backtest(pead_w["filed"], adj[stocks])

    ff = fama_french()
    perf_rows, expo = [], {}
    for name, bt in bts.items():
        g, n = portfolio.perf(bt["gross"]), portfolio.perf(bt["net"], bt["turnover"])
        beta_mkt = np.cov(bt["net"], mret.reindex(bt.index).fillna(0))[0, 1] / mret.reindex(bt.index).var()
        perf_rows.append({"portfolio": name, "sharpe_gross": g["sharpe"], "sharpe_net": n["sharpe"],
                          "ann_ret_net_pct": n["ann_return_pct"], "ann_vol_pct": n["ann_vol_pct"],
                          "max_dd_net_pct": n["max_drawdown_pct"], "turnover_x_per_year": n["turnover_ann"],
                          "beta_SPY": beta_mkt})
        expo[name] = portfolio.factor_regression(bt["net"], ff)
    perf_tab = pd.DataFrame(perf_rows).set_index("portfolio")
    perf_tab.to_csv(RESULTS / "portfolio_performance.csv")
    expo_tab = pd.DataFrame({k: v["coef"] for k, v in expo.items()}).T
    expo_t = pd.DataFrame({k: v["t"] for k, v in expo.items()}).T
    expo_tab.to_csv(RESULTS / "factor_exposures_coef.csv")
    expo_t.to_csv(RESULTS / "factor_exposures_tstat.csv")
    # cost sensitivity and sub-periods for the combined books
    sens = {}
    for name, wts in {"EW_sleeves": ew_w, "RiskParity_sleeves": erc_w, "MeanVariance_LW": mv_w,
                      "PEAD_8K_timing": pead_w["release"], "PEAD_10Q_timing": pead_w["filed"]}.items():
        sens[name] = {f"{c:g}bps": portfolio.perf(portfolio.backtest(wts, adj[stocks], cost_bps=c)["net"])["sharpe"]
                      for c in [0, 5, 10, 20]}
    sens = pd.DataFrame(sens).T
    sens.to_csv(RESULTS / "cost_sensitivity_sharpe.csv")
    subp = {}
    for name, bt in bts.items():
        subp[name] = {p: portfolio.perf(bt["net"].loc[a:b])["sharpe"]
                      for p, (a, b) in {"2014-2019": ("2014", "2019"), "2020-2026": ("2020", "2026")}.items()}
    subp = pd.DataFrame(subp).T
    subp.to_csv(RESULTS / "portfolio_sharpe_subperiods.csv")
    erc_tab = pd.DataFrame(erc_b, index=sleeves).T
    navs = pd.DataFrame({k: (1 + v["net"]).cumprod() for k, v in bts.items()})
    navs["Pairs"] = (1 + pr["net"]).cumprod().reindex(navs.index).ffill()
    navs.to_csv(RESULTS / "nav_net.csv")

    # ------------------------------------------------------------ 7. write-up
    print("[7/7] figures and summary", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for c in ["EW_sleeves", "RiskParity_sleeves", "MeanVariance_LW", "PEAD_8K_timing", "PEAD_10Q_timing", "Pairs"]:
        ax[0].plot(navs.index, navs[c], label=c, lw=1.4)
    for c in sleeves:
        ax[0].plot(navs.index, navs[c], label=c, lw=0.8, alpha=0.6, ls="--")
    ax[0].axhline(1, color="grey", lw=0.6)
    ax[0].set_title(f"Net-of-cost NAV ({COST_BPS:g} bps one-way)")
    ax[0].legend(fontsize=7, ncol=2)
    for k in ["MOM", "VALUE", "ROA", "SUE", "SUE_10Q_timing", "EAR"]:
        ax[1].plot(decay.columns, decay.loc[k], marker="o", label=k)
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_xlabel("month after formation")
    ax[1].set_title("Rank IC decay")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "nav_and_ic_decay.png", dpi=130)

    cov_year = q.pop("coverage_by_year")
    cov_year.to_csv(RESULTS / "price_coverage_by_year.csv")
    summary = {"data_quality": q, "pairs": {**pairs_summary, "perf_gross": pp["gross"], "perf_net": pp["net"]},
               "mv_avg_shrinkage": float(mv_diag["shrinkage"].mean()),
               "mv_avg_ex_ante_vol_ann": float(mv_diag["ex_ante_vol_ann"].mean()),
               "mv_avg_gross": float(mv_diag["gross"].mean()), "mv_avg_names": float(mv_diag["names"].mean()),
               "mv_solver_status": mv_diag["status"].value_counts().to_dict(),
               "erc_avg_sleeve_weights": erc_tab.loc["2015":].mean().round(3).to_dict(),
               "runtime_minutes": round((time.time() - t0) / 60, 1)}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    tables = ["# Results (generated by `uv run altdata`)\n",
              "## Signal evaluation (monthly, sector-neutral ranks)\n", _md(ic_tab),
              "\n## Earnings event study: market-adjusted log return %, by SUE quintile\n", _md(es_tab, "{:.2f}"),
              "\n## Rank IC decay (mean rank IC with the return of month h)\n", _md(decay),
              "\n## Rank IC by sub-period\n", _md(sub),
              "\n## Quintile returns, annualised % (equal-weighted)\n", _md(pd.DataFrame(quant).T, "{:.2f}"),
              "\n## Portfolios\n", _md(perf_tab, "{:.2f}"),
              "\n## Net Sharpe by sub-period\n", _md(subp, "{:.2f}"),
              "\n## Net Sharpe vs one-way cost\n", _md(sens, "{:.2f}"),
              "\n## Fama-French 5 + momentum exposures of net returns (coef)\n", _md(expo_tab, "{:.3f}"),
              "\n## ... t-stats (Newey-West, 5 lags)\n", _md(expo_t, "{:.2f}"),
              "\n## Pairs\n", "```\n" + json.dumps(summary["pairs"], indent=2, default=str) + "\n```",
              "\n## Data quality\n", "```\n" + json.dumps(q, indent=2, default=str) + "\n```",
              "\n## Price coverage by year\n", _md(cov_year)]
    (RESULTS / "RESULTS.md").write_text("\n".join(tables) + "\n")
    print(f"done in {summary['runtime_minutes']} min", flush=True)


if __name__ == "__main__":
    main()
