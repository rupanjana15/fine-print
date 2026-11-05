"""Polite, cached, resumable fetcher for the ToS;DR public API.

Every response is cached as JSON under data/raw/tosdr/, so re-running skips anything already
downloaded — you can stop it (Ctrl+C) or hit rate limits and simply run it again.

Notes on the API (checked 2026-11-04; the published OpenAPI spec is incomplete):
- /point/v1?case_id=X&page=N  returns all points for a case *with* quote_text/quote_start/
  quote_end and status (approved / declined / ...). This is our main source.
- /service/v3?id=X            returns approved points but WITHOUT quotes.
- /service/v3?page=N          lists services, 500 per page (~10k services).
- Rate limit: headers say 5 req / 5 s, but bursts get HTTP 429 "You are doing this too much".

Usage:
  uv run python -m fineprint.data.fetch_tosdr points     # all points, by case (main data)
  uv run python -m fineprint.data.fetch_tosdr documents  # full text + service_id (run after points)
  uv run python -m fineprint.data.fetch_tosdr services   # service list (names, urls, ratings)
  uv run python -m fineprint.data.fetch_tosdr all
"""

import argparse
import json
import time
from pathlib import Path

import requests

from fineprint.config import data_path, load_config


class RateLimitedClient:
    """GET JSON with a fixed delay between requests and exponential backoff on 429/5xx."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg["user_agent"]
        self._last_request = 0.0

    def get(self, path: str, params: dict) -> dict | None:
        for attempt in range(self.cfg["max_retries"]):
            wait = self.cfg["seconds_between_requests"] - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

            try:
                resp = self.session.get(f"{self.cfg['base_url']}{path}", params=params, timeout=60)
            except requests.RequestException as e:
                self._backoff(attempt, f"network error: {e}")
                continue

            if resp.status_code == 200:
                if resp.headers.get("X-RateLimit-Remaining") == "0":
                    time.sleep(float(resp.headers.get("X-RateLimit-Reset", 5)))
                return resp.json()
            # 404 = unknown resource; 400 "Invalid page number ... <= 0" = case with no points.
            if resp.status_code == 404 or (
                resp.status_code == 400 and "Invalid page number" in resp.text
            ):
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                self._backoff(attempt, f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()

        raise RuntimeError(f"Gave up on {path} {params} after {self.cfg['max_retries']} tries")

    def _backoff(self, attempt: int, reason: str) -> None:
        delay = min(self.cfg["backoff_base_seconds"] * 2**attempt, self.cfg["max_backoff_seconds"])
        print(f"  {reason} — sleeping {delay:.0f}s")
        time.sleep(delay)


def cached_get(client: RateLimitedClient, cache_file: Path, path: str, params: dict) -> dict | None:
    """Return the cached response if present, else fetch and cache it.

    Missing resources are cached as {"_missing": true} so they are not re-requested.
    """
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return None if data.get("_missing") else data

    data = client.get(path, params)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data if data is not None else {"_missing": True}))
    return data


def fetch_points_by_case(client: RateLimitedClient, cfg: dict, out: Path) -> None:
    """Scan case ids, fetching every page of points for each case."""
    empty_streak = 0
    total_points = 0

    for case_id in range(1, cfg["max_case_id"] + 1):
        first = cached_get(client, out / f"case_{case_id}_page_1.json", "/point/v1/",
                           {"case_id": case_id, "page": 1})
        points = (first or {}).get("points") or []
        if not points:
            empty_streak += 1
            if empty_streak >= cfg["stop_after_empty_cases"]:
                print(f"Stopping at case {case_id}: {empty_streak} empty ids in a row")
                break
            continue
        empty_streak = 0

        last_page = first.get("page", {}).get("end", 1)
        for page in range(2, last_page + 1):
            data = cached_get(client, out / f"case_{case_id}_page_{page}.json", "/point/v1/",
                              {"case_id": case_id, "page": page})
            points += (data or {}).get("points") or []

        total_points += len(points)
        title = points[0]["case"]["title"][:60]
        print(f"case {case_id:>4}: {len(points):>5} points ({last_page} pages) "
              f"| total {total_points:,} | {title}")


def fetch_services(client: RateLimitedClient, out: Path) -> None:
    """Fetch the paginated service list."""
    first = cached_get(client, out / "page_1.json", "/service/v3/", {"page": 1})
    last_page = first["page"]["end"]
    for page in range(2, last_page + 1):
        cached_get(client, out / f"page_{page}.json", "/service/v3/", {"page": page})
        print(f"services page {page}/{last_page}")
    print(f"Service list complete: {first['page']['total']:,} services")


def approved_document_ids(points_dir: Path) -> list[int]:
    """Document ids referenced by human-approved points in the points cache."""
    ids = set()
    for f in points_dir.glob("case_*_page_*.json"):
        for p in json.loads(f.read_text()).get("points") or []:
            if p.get("status") == "approved" and p.get("document_id"):
                ids.add(p["document_id"])
    return sorted(ids)


def fetch_documents(client: RateLimitedClient, points_dir: Path, out: Path) -> None:
    """Fetch full text + service_id for every document behind an approved point.

    Needed because ~40% of points have no source URL: service_id is what lets us split by company,
    and the full text is where harmless (negative) chunks come from.
    """
    ids = approved_document_ids(points_dir)
    print(f"{len(ids):,} documents referenced by approved points")
    for i, doc_id in enumerate(ids, 1):
        # /document/v2 returns "404 Route Not Found" despite the spec; v1 works and wraps the
        # document (text, service_id, ...) in a "parameters" key.
        cached_get(client, out / f"doc_{doc_id}.json", "/document/v1/", {"id": doc_id})
        if i % 50 == 0:
            print(f"documents {i:,}/{len(ids):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["points", "documents", "services", "all"])
    args = parser.parse_args()

    cfg = load_config("data")["tosdr"]
    client = RateLimitedClient(cfg)
    raw = data_path("raw") / "tosdr"

    if args.what in ("services", "all"):
        fetch_services(client, raw / "services")
    if args.what in ("points", "all"):
        fetch_points_by_case(client, cfg, raw / "points_by_case")
    if args.what in ("documents", "all"):
        fetch_documents(client, raw / "points_by_case", raw / "documents")


if __name__ == "__main__":
    main()
