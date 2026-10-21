"""Point-in-time S&P 500 membership rebuilt from a pinned Wikipedia revision.

The revision lists today's constituents and a table of index changes
(date, added ticker, removed ticker). Walking the changes backwards from the
snapshot gives the member set on any past date. Membership after the snapshot
date is held at the snapshot (stale but never forward-looking).
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .config import RAW, WIKI_REVID, WIKI_USER_AGENT


def _norm(t) -> str | None:
    if not isinstance(t, str) or not t.strip():
        return None
    return t.strip().upper().replace(".", "-")  # BRK.B -> BRK-B (Yahoo style)


def fetch_wikipedia(revid: int = WIKI_REVID) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = RAW / f"wiki_sp500_{revid}.html"
    if not path.exists():
        r = requests.get("https://en.wikipedia.org/w/index.php", params={"oldid": revid},
                         headers={"User-Agent": WIKI_USER_AGENT}, timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    tables = pd.read_html(io.StringIO(path.read_text(encoding="utf-8")))
    current = tables[0]
    changes = tables[1]
    changes.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]
    changes["date"] = pd.to_datetime(changes["date"], format="mixed", errors="coerce")
    changes["added"] = changes["added"].map(_norm)
    changes["removed"] = changes["removed"].map(_norm)
    current = current.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "gics_sector",
                                      "CIK": "cik", "Date added": "date_added"})
    current["ticker"] = current["ticker"].map(_norm)
    # Snapshot date: latest change in the table (revision timestamp is 2026-06-30).
    snapshot = "2026-06-30"
    return current, changes.dropna(subset=["date"]), snapshot


def membership_intervals(current: list[str], changes: pd.DataFrame, snapshot: str,
                         start: str) -> tuple[pd.DataFrame, dict]:
    """Reverse-walk the change table. Returns one row per (ticker, start, end) interval.

    `end` is exclusive and NaT for names still in the index at the snapshot.
    """
    members = set(current)
    open_end: dict[str, pd.Timestamp | None] = {t: None for t in members}
    rows = []
    issues = {"added_not_member": 0, "removed_already_member": 0}
    snap = pd.Timestamp(snapshot)
    start_ts = pd.Timestamp(start)
    for _, ch in changes[changes["date"] <= snap].sort_values("date", ascending=False).iterrows():
        d = ch["date"]
        if d < start_ts:
            break
        add, rem = ch["added"], ch["removed"]
        add = add if isinstance(add, str) else None
        rem = rem if isinstance(rem, str) else None
        if add:
            if add in members:
                members.discard(add)
                rows.append((add, d, open_end.pop(add)))
            else:
                issues["added_not_member"] += 1
        if rem:
            if rem in members:
                issues["removed_already_member"] += 1
            else:
                members.add(rem)
                open_end[rem] = d
    for t in members:
        rows.append((t, pd.NaT, open_end.get(t)))
    iv = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    iv["start"] = iv["start"].fillna(start_ts)
    iv["end"] = pd.to_datetime(iv["end"])
    return iv.sort_values(["ticker", "start"]).reset_index(drop=True), issues


def member_mask(intervals: pd.DataFrame, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Boolean (date x ticker) frame: True when the ticker is an index member on that date."""
    mask = pd.DataFrame(False, index=dates, columns=pd.Index(tickers))
    for t, s, e in intervals.itertuples(index=False):
        if t not in mask.columns:
            continue
        sel = dates >= s
        if pd.notna(e):
            sel &= dates < e
        mask.loc[sel, t] = True
    return mask
