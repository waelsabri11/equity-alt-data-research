"""Paths, study window and every fixed research choice in one place.

All parameters below were set before looking at backtest results and are not tuned.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"            # downloaded files (never committed)
CACHE = DATA / "edgar_cache"  # gzip-compressed SEC responses (never committed)
INTERIM = DATA / "interim"    # cleaned parquet panels
RESULTS = ROOT / "results"

# Read-only fallback: the EDGAR cache of the llama3-finreport-extraction project
# (same URL hashing). Saves re-downloading company facts that are already on disk.
SEED_CACHE = Path(os.environ.get(
    "EDGAR_SEED_CACHE", Path.home() / "Projects/llama3-finreport-extraction/data/cache"))

# SEC fair-access policy: declare who you are, stay under 10 requests/second.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "equity-alt-data-research waelsabri4@gmail.com")
WIKI_USER_AGENT = SEC_USER_AGENT

# Pinned Wikipedia revision (2026-06-30) that still holds the S&P 500 change table.
WIKI_REVID = 1361794875

PRICE_START = "2012-01-01"
PRICE_END = "2026-10-01"      # exclusive
EVAL_START = "2014-01-01"     # first rebalance month (needs 1y+ of history before)
MARKET = "SPY"

SEED = 7

# Costs (one-way, applied to traded notional) and short borrow fee.
COST_BPS = 10.0
BORROW_BPS_PER_YEAR = 30.0

# Pairs trading
PAIRS_FORMATION = 252
PAIRS_TRADING = 126
PAIRS_PREFILTER = 20          # lowest-SSD candidates per industry tested for cointegration
PAIRS_MAX = 20                # pairs traded per window
PAIRS_PVALUE = 0.05
Z_ENTRY, Z_EXIT, Z_STOP = 2.0, 0.5, 4.0
PAIRS_COST_BPS = 10.0         # per leg, one-way

# Portfolio construction
COV_WINDOW = 252
NAME_CAP = 0.02               # |w_i| cap in the optimiser (gross 2 = 1 long + 1 short)
MV_RISK_AVERSION = 20.0
ASSUMED_IC = 0.03             # Grinold alpha scaling: alpha = IC * vol * z
