"""
IntelliJob — Phase 1 dataset collection
====================================

Pulls a static, frozen "research dataset" of UK software / data / IT job
specifications from the Adzuna Jobs API. This is the *offline seed* that
satisfies the dissertation spec:

    Phase A (Offline preparation — done once):
        - Static, frozen, uncurated research dataset of UK software
          engineering job specs is seeded into PostgreSQL.

We hit the Adzuna API once, save the raw responses to a CSV, and that file
becomes the canonical input to the embedding pipeline in Phase B.

Search coverage: every career pathway in api/services/pathways.py gets at
least one targeted query (embedded, mobile, game, security, QA, ...), not
just the generic "software engineer" titles — otherwise those careers can
never be recommended, however good the matching is.

Run with:
    intellijob-backend/venv/Scripts/python.exe scripts/seed_jobs.py
    intellijob-backend/venv/Scripts/python.exe scripts/seed_jobs.py --dry-run

After re-seeding, rebuild the vectors so the matcher/pathway engine picks
up the new jobs:
    intellijob-backend/venv/Scripts/python.exe scripts/embed_jobs.py

Output:
    intellijob-backend/data/uk_software_jobs.csv
    intellijob-backend/data/MANIFEST.json   (row count, sha256, queries, ts)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

import requests
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Project root is two levels up from this script:
#   scripts/seed_jobs.py  →  intellijob-backend/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
    sys.exit(
        "ERROR: ADZUNA_APP_ID / ADZUNA_APP_KEY not found.\n"
        f"Add them to {PROJECT_ROOT / '.env'} and re-run."
    )

# Search queries, grouped by career pathway (see api/services/pathways.py).
# Each pathway gets at least one targeted query so the dataset is not
# dominated by generic "software engineer" postings — an embedded or mobile
# resume can only be recommended if the seed actually contains those roles.
SEARCH_QUERIES: list[str] = [
    # software-engineering (generic catch-all)
    "software engineer",
    "software developer",
    # ml-ai
    "data scientist",
    "machine learning engineer",
    "ai engineer",
    "mlops engineer",
    "nlp engineer",
    "computer vision engineer",
    # data-eng
    "data engineer",
    "etl developer",
    "big data engineer",
    # data-analytics
    "data analyst",
    "business intelligence analyst",
    "business analyst",
    # devops
    "devops engineer",
    "cloud engineer",
    "site reliability engineer",
    "platform engineer",
    "infrastructure engineer",
    "kubernetes engineer",
    "release engineer",
    "linux system administrator",
    "linux engineer",
    # backend
    "backend developer",
    "java developer",
    ".net developer",
    "python developer",
    "node.js developer",
    "golang developer",
    "c++ developer",
    "php developer",
    # frontend
    "frontend developer",
    "react developer",
    "vue developer",
    "angular developer",
    "web developer",
    "ui engineer",
    # fullstack
    "full stack developer",
    "fullstack developer",
    # mobile
    "mobile developer",
    "ios developer",
    "android developer",
    "react native developer",
    "flutter developer",
    # game
    "game developer",
    "unity developer",
    "unreal developer",
    "graphics programmer",
    # embedded / hardware
    "embedded software engineer",
    "firmware engineer",
    "hardware engineer",
    "fpga engineer",
    "iot engineer",
    "robotics engineer",
    "control systems engineer",
    # security
    "cyber security analyst",
    "security engineer",
    "penetration tester",
    "application security engineer",
    "information security analyst",
    # qa
    "qa automation engineer",
    "sdet",
    "test engineer",
    # architecture
    "solution architect",
    "technical architect",
    "enterprise architect",
    "cloud architect",
    # delivery
    "product manager",
    "project manager",
    "program manager",
    "scrum master",
    "technical account manager",
]

# UK locations — a spread of IT hubs gives geographic diversity.
# (Adzuna's `where` accepts a city or region name.)
LOCATIONS: list[str] = [
    "London",
    "Manchester",
    "Edinburgh",
    "Glasgow",
    "Bristol",
]

# Max results to request per query. Adzuna caps each page at 50 and
# uses 1 API call per page. Two pages per (query, location) roughly
# doubles the per-role coverage without hammering the free tier.
RESULTS_PER_PAGE = 50
MAX_PAGES = 2  # 2 pages × 50 = up to 100 results per (query, location)

OUTPUT_CSV = PROJECT_ROOT / "data" / "uk_software_jobs.csv"
MANIFEST_JSON = PROJECT_ROOT / "data" / "MANIFEST.json"
ADZUNA_ENDPOINT = "https://api.adzuna.com/v1/api/jobs/gb/search/{page}"

CSV_FIELDS: list[str] = [
    "id",
    "title",
    "company_display_name",
    "location_display",
    "location_area",
    "salary_min",
    "salary_max",
    "salary_is_predicted",
    "contract_type",
    "contract_time",
    "category_label",
    "created",
    "description",
    "redirect_url",
    "search_query",
    "search_location",
]

# Polite delay between API calls. Adzuna's free tier doesn't publish a
# documented rate limit; 0.3s is comfortably below what their public
# examples use.
API_DELAY_SECONDS = 0.3

# Retry policy for transient failures (429 / 5xx / network).
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5  # multiplied by attempt number


# --------------------------------------------------------------------------- #
# HTML stripping
# --------------------------------------------------------------------------- #


class _HTMLStripper(HTMLParser):
    """Convert Adzuna's HTML job description to clean text.

    Drops tags, decodes the few entities Adzuna uses (&amp;, &nbsp;,
    &#x27;), collapses runs of whitespace, and trims. Keeps <br> and
    </p> as line breaks so paragraph structure survives. Strips the
    body of <script>/<style> entirely (HTMLParser does not do this
    by default).
    """

    BLOCK_TAGS: ClassVar[set[str]] = {"p", "br", "li", "div", "h1", "h2", "h3", "h4", "tr"}
    # Tags whose textual content we must never let through.
    DROP_TAGS: ClassVar[set[str]] = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._drop_depth = 0  # >0 means we are inside <script>/<style>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if tag_l in self.DROP_TAGS:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag_l in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in self.DROP_TAGS:
            if self._drop_depth:
                self._drop_depth -= 1
            return
        if self._drop_depth:
            return
        if tag_l in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth:
            return
        self._chunks.append(data)

    def handle_entityref(self, name: str) -> None:  # legacy &name; refs
        if self._drop_depth:
            return
        # Delegate to the default entity decoding by routing through
        # the data handler.
        self._chunks.append(html.unescape(f"&{name};"))

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        # NBSP (U+00A0) and friends should collapse like normal space.
        raw = raw.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
        # Split into lines, collapse internal whitespace runs, strip,
        # drop empties. Preserves block-tag line breaks set above.
        lines = []
        for ln in raw.splitlines():
            ln = " ".join(ln.split())  # collapses any run of whitespace
            if ln:
                lines.append(ln)
        return "\n".join(lines)


def html_to_text(html: str) -> str:
    """Strip HTML tags and normalise whitespace. Returns '' for falsy input."""
    if not html:
        return ""
    stripper = _HTMLStripper()
    try:
        stripper.feed(html)
        stripper.close()
    except Exception:  # noqa: BLE001 - any parser defect -> crude strip fallback
        # If Adzuna ever sends something malformed, fall back to a crude
        # strip rather than crashing the whole seed run.
        return " ".join(html.split())
    return stripper.get_text()


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def fetch_page(
    page: int, query: str, location: str
) -> list[dict[str, Any]]:
    """Fetch a single page of Adzuna results for one (query, location).

    Retries on 429 / 5xx / network errors with linear backoff.
    """
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": query,
        "where": location,
        "content-type": "application/json",
    }
    url = ADZUNA_ENDPOINT.format(page=page)

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            if (
                response.status_code in (429, 500, 501, 502, 503, 504)
                and attempt < MAX_RETRIES
            ):
                # Transient — back off and retry.
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"   (retry {attempt}/{MAX_RETRIES - 1} "
                    f"after {wait:.1f}s — HTTP {response.status_code})",
                    flush=True,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.HTTPError:
            # Non-transient HTTP error: don't retry, surface immediately.
            raise
        except requests.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"   (retry {attempt}/{MAX_RETRIES - 1} "
                    f"after {wait:.1f}s — {e})",
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise

    # Should be unreachable, but keep type-checkers happy.
    if last_exc:
        raise last_exc
    return []


def to_row(result: dict[str, Any], query: str, location: str) -> dict[str, Any]:
    """Flatten an Adzuna result JSON into the CSV row shape."""
    company = result.get("company") or {}
    location_ = result.get("location") or {}
    area = location_.get("area") or []
    # Adzuna returns id as a JSON int in some cases and str in others.
    # Normalise to str so the CSV and dedup logic see a consistent type.
    raw_id = result.get("id", "")
    return {
        "id": "" if raw_id is None else str(raw_id),
        "title": (result.get("title") or "").strip(),
        "company_display_name": (company.get("display_name") or "").strip(),
        "location_display": (location_.get("display_name") or "").strip(),
        "location_area": ", ".join(area) if isinstance(area, list) else str(area),
        "salary_min": "" if result.get("salary_min") is None else result["salary_min"],
        "salary_max": (
            "" if result.get("salary_max") is None else result["salary_max"]
        ),
        "salary_is_predicted": result.get("salary_is_predicted", ""),
        "contract_type": result.get("contract_type", ""),
        "contract_time": result.get("contract_time", ""),
        "category_label": (result.get("category") or {}).get("label", ""),
        "created": result.get("created", ""),
        "description": html_to_text(result.get("description") or ""),
        "redirect_url": result.get("redirect_url", ""),
        "search_query": query,
        "search_location": location,
    }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    rows: list[dict[str, Any]],
    csv_path: Path,
    manifest_path: Path,
    api_calls: int,
    queries_used: list[str],
    locations_used: list[str],
) -> None:
    """Write a small JSON manifest next to the CSV so the dataset is citable."""
    manifest = {
        "csv_filename": csv_path.name,
        "csv_sha256": _sha256_of_file(csv_path) if csv_path.exists() else None,
        "row_count": len(rows),
        "row_count_unique": len({str(r.get("id", "")) for r in rows if r.get("id")}),
        "queries": queries_used,
        "locations": locations_used,
        "results_per_page": RESULTS_PER_PAGE,
        "max_pages": MAX_PAGES,
        "api_calls_used": api_calls,
        "source": "Adzuna Jobs API (https://developer.adzuna.com/)",
        "country": "GB",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": 1,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def collect_rows() -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    """Hit Adzuna for every (location × query) and return deduped rows.

    Returns (rows, api_calls, stats) where stats has per-status counts
    for logging.
    """
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    api_calls = 0
    stats = {"fetched": 0, "deduped": 0, "skipped_no_id": 0, "errors": 0}

    for location in LOCATIONS:
        for query in SEARCH_QUERIES:
            print(f"  -> {query!r:35s} | {location!r:12s}", end=" ", flush=True)
            query_added = 0
            for page in range(1, MAX_PAGES + 1):
                try:
                    page_results = fetch_page(page=page, query=query, location=location)
                except requests.HTTPError as e:
                    print(f"HTTP {e.response.status_code} — skipping")
                    stats["errors"] += 1
                    break
                except requests.RequestException as e:
                    print(f"network error: {e} — skipping")
                    stats["errors"] += 1
                    break
                api_calls += 1
                stats["fetched"] += len(page_results)

                page_added = 0
                for result in page_results:
                    rid = str(result.get("id", ""))
                    if not rid:
                        stats["skipped_no_id"] += 1
                        continue
                    if rid in seen_ids:
                        stats["deduped"] += 1
                        continue
                    seen_ids.add(rid)
                    rows.append(to_row(result, query=query, location=location))
                    page_added += 1
                query_added += page_added

                if len(page_results) < RESULTS_PER_PAGE:
                    break  # fewer than a full page => no more results
                time.sleep(API_DELAY_SECONDS)
            print(
                f"-> {query_added:3d} new ({len(rows):4d} total, "
                f"{api_calls} API calls)"
            )

    return rows, api_calls, stats


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IntelliJob Phase A — Adzuna seeder.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch from Adzuna but do not write CSV or manifest.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_JSON,
        help=f"Manifest JSON path (default: {MANIFEST_JSON})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("IntelliJob Phase A — Adzuna seeder")
    print(f"  queries   : {len(SEARCH_QUERIES)}")
    print(f"  locations : {len(LOCATIONS)}")
    print(f"  pages     : {MAX_PAGES} × {RESULTS_PER_PAGE} = "
          f"{len(SEARCH_QUERIES) * len(LOCATIONS) * MAX_PAGES} max API calls")
    if args.dry_run:
        print("  mode      : DRY RUN (no files will be written)")
    print()

    rows, api_calls, stats = collect_rows()

    print()
    print(f"Fetched {stats['fetched']} raw results")
    print(f"  - {stats['deduped']} duplicates dropped")
    print(f"  - {stats['skipped_no_id']} skipped (no id)")
    print(f"  - {stats['errors']} (query, location) pairs failed")
    print(f"Wrote {len(rows)} unique jobs (using {api_calls} API calls)")

    if args.dry_run:
        print("\n--dry-run set; not writing CSV or manifest.")
        return 0

    write_csv(rows, args.output)
    print(f"  CSV  -> {args.output}")

    write_manifest(
        rows=rows,
        csv_path=args.output,
        manifest_path=args.manifest,
        api_calls=api_calls,
        queries_used=SEARCH_QUERIES,
        locations_used=LOCATIONS,
    )
    print(f"  MFT  -> {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
