"""SEC EDGAR client: rate-limited, retried, gzip-cached on disk.

Adapted from llama3-finreport-extraction/finx/edgar.py. Responses are cached
compressed; the finreport cache is read as a fallback so nothing is fetched twice.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import requests

from .config import CACHE, SEC_USER_AGENT, SEED_CACHE

_MIN_INTERVAL = 0.125  # 8 requests/second, under the SEC limit of 10


class Edgar:
    def __init__(self, cache_dir: Path = CACHE, seed_dir: Path | None = SEED_CACHE):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed_dir = seed_dir if seed_dir and seed_dir.exists() else None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0
        self.n_requests = 0

    @staticmethod
    def _key(url: str) -> str:
        return hashlib.sha1(url.encode()).hexdigest()

    def get(self, url: str) -> str | None:
        h = self._key(url)
        path = self.cache_dir / h[:2] / f"{h}.gz"
        if path.exists():
            text = gzip.decompress(path.read_bytes()).decode("utf-8")
            return text or None  # empty = cached 404
        if self.seed_dir is not None:
            seed = self.seed_dir / h[:2] / h
            if seed.exists():
                return seed.read_text(encoding="utf-8") or None
        for attempt in range(4):
            wait = _MIN_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                self._store(path, "")
                return None
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            self._store(path, r.text)
            return r.text
        return None

    @staticmethod
    def _store(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=6))

    def get_json(self, url: str) -> dict | None:
        text = self.get(url)
        return json.loads(text) if text else None

    def tickers(self) -> dict[str, int]:
        data = self.get_json("https://www.sec.gov/files/company_tickers.json")
        return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}

    def submissions(self, cik: int) -> dict | None:
        """Company filings, with the paginated older history merged in."""
        sub = self.get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if sub is None:
            return None
        recent = sub["filings"]["recent"]
        for extra in sub["filings"].get("files", []):
            page = self.get_json(f"https://data.sec.gov/submissions/{extra['name']}")
            if page:
                n = len(page["form"])
                for k in recent:
                    recent[k].extend(page.get(k, [""] * n))
        return sub

    def company_facts(self, cik: int) -> dict | None:
        return self.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
