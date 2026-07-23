"""
IntelliJob — typed reader for the Phase A Adzuna job dataset.

This module is the single source of truth for *how a job row looks once
it's been written to disk* and *how to validate it before downstream
consumers (embeddings, RAG, UI) trust it*.

It deliberately has no Django, no spaCy, no pgvector — just stdlib — so
that:

  * scripts/seed_jobs.py (writer) and the future embedding pipeline
    (reader) share one shape,
  * tests run without a database or model registry,
  * the loader can be reused unchanged when we later switch the storage
    backend.

Public API:
    JobRow(...)              - typed row dataclass
    read_jobs_csv(path)      - parse CSV -> list[JobRow]
    validate_rows(rows)      - produce a ValidationReport
    ValidationReport         - dataclass with counts + flagged row indices

CSV schema (matches scripts/seed_jobs.py CSV_FIELDS, schema_version 1):
    id, title, company_display_name, location_display, location_area,
    salary_min, salary_max, salary_is_predicted, contract_type,
    contract_time, category_label, created, description, redirect_url,
    search_query, search_location
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

#: CSV column order, frozen at schema_version=1. Bumping the schema
#: version (in seed_jobs.MANIFEST) is the way to evolve this list.
CSV_COLUMNS: tuple[str, ...] = (
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
)

#: Columns we want as numbers. Empty string -> None (a real Adzuna
#: "salary not given" signal, not zero).
INT_COLUMNS: tuple[str, ...] = ("salary_min", "salary_max")
BOOL_COLUMNS: tuple[str, ...] = ("salary_is_predicted",)

#: Minimum description length, below which a row is flagged as
#: low-quality. Adzuna truncates some listings to ~200 chars; those
#: are basically useless for embedding.
MIN_DESCRIPTION_CHARS = 200


# --------------------------------------------------------------------------- #
# Row type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobRow:
    """One job spec, parsed and lightly typed.

    All string fields are stripped; missing strings are normalised to ''.
    Numeric fields are ``int | None`` (None = "Adzuna did not return a
    value", not zero).
    """

    id: str
    title: str
    company_display_name: str
    location_display: str
    location_area: str
    salary_min: int | None
    salary_max: int | None
    salary_is_predicted: bool | None
    contract_type: str
    contract_time: str
    category_label: str
    created: str  # ISO-ish, kept as string to avoid TZ parsing bugs
    description: str
    redirect_url: str
    search_query: str
    search_location: str

    def is_uk(self) -> bool:
        """Heuristic UK check: non-empty UK-sounding location.

        We don't want to hard-fail rows for a missing location, but
        callers can use this to filter before embedding.
        """
        loc = (self.location_display or "").lower()
        if not loc:
            return False
        uk_markers = (
            "london", "manchester", "edinburgh", "birmingham", "glasgow",
            "bristol", "leeds", "liverpool", "newcastle", "nottingham",
            "sheffield", "cardiff", "belfast", "cambridge", "oxford",
            "reading", "uk", "united kingdom", "england", "scotland",
            "wales", "northern ireland",
        )
        return any(marker in loc for marker in uk_markers)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _coerce_int(value: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        # Adzuna occasionally returns floats like "45000.0" — accept those.
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _coerce_bool(value: str) -> bool | None:
    if value is None or value == "":
        return None
    v = value.strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    return None


def _row_from_mapping(raw: dict[str, str]) -> JobRow:
    """Build a JobRow from a csv.DictReader mapping, tolerating missing keys."""
    # csv.DictReader gives us a dict keyed by the file's header row.
    # We look up by CSV_COLUMNS so a slightly out-of-order file still loads.
    def get(col: str) -> str:
        for k, v in raw.items():
            if k is not None and k.strip() == col:
                return (v or "").strip()
        return ""

    return JobRow(
        id=get("id"),
        title=get("title"),
        company_display_name=get("company_display_name"),
        location_display=get("location_display"),
        location_area=get("location_area"),
        salary_min=_coerce_int(get("salary_min")),
        salary_max=_coerce_int(get("salary_max")),
        salary_is_predicted=_coerce_bool(get("salary_is_predicted")),
        contract_type=get("contract_type"),
        contract_time=get("contract_time"),
        category_label=get("category_label"),
        created=get("created"),
        description=get("description"),
        redirect_url=get("redirect_url"),
        search_query=get("search_query"),
        search_location=get("search_location"),
    )


def read_jobs_csv(path: str | Path) -> list[JobRow]:
    """Read the Phase A CSV and return typed JobRow objects.

    Does NOT validate — just parses. Run :func:`validate_rows` afterwards
    if you want a quality report.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job CSV not found: {path}")

    rows: list[JobRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(_row_from_mapping(raw))
    return rows


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass
class ValidationReport:
    """Quality report for a loaded dataset.

    ``flagged_indices`` indexes into the input list, so callers can
    filter or log without re-walking the data.
    """

    total: int = 0
    missing_id: int = 0
    missing_title: int = 0
    missing_description: int = 0
    short_description: int = 0
    non_uk_location: int = 0
    duplicate_id: int = 0
    flagged_indices: list[int] = field(default_factory=list)

    @property
    def clean_count(self) -> int:
        return self.total - len(set(self.flagged_indices))

    def summary(self) -> str:
        return (
            f"total={self.total} clean={self.clean_count} "
            f"missing_id={self.missing_id} missing_title={self.missing_title} "
            f"missing_description={self.missing_description} "
            f"short_description={self.short_description} "
            f"non_uk_location={self.non_uk_location} "
            f"duplicate_id={self.duplicate_id}"
        )


def validate_rows(
    rows: Iterable[JobRow],
    *,
    require_uk: bool = False,
) -> ValidationReport:
    """Produce a :class:`ValidationReport` for a sequence of JobRow.

    A row is "flagged" if it has any of:
      * empty id
      * empty title
      * empty description
      * description shorter than MIN_DESCRIPTION_CHARS
      * non-UK location (only checked when ``require_uk=True``)
      * duplicate id (only the second-and-later occurrence is flagged)
    """
    report = ValidationReport()
    seen_ids: set[str] = set()

    for i, row in enumerate(rows):
        report.total += 1
        bad = False
        if not row.id:
            report.missing_id += 1
            bad = True
        if not row.title:
            report.missing_title += 1
            bad = True
        if not row.description:
            report.missing_description += 1
            bad = True
        elif len(row.description) < MIN_DESCRIPTION_CHARS:
            report.short_description += 1
            bad = True
        if require_uk and not row.is_uk():
            report.non_uk_location += 1
            bad = True
        if row.id:
            if row.id in seen_ids:
                report.duplicate_id += 1
                bad = True
            else:
                seen_ids.add(row.id)

        if bad:
            report.flagged_indices.append(i)

    return report


__all__ = [
    "CSV_COLUMNS",
    "INT_COLUMNS",
    "BOOL_COLUMNS",
    "MIN_DESCRIPTION_CHARS",
    "JobRow",
    "ValidationReport",
    "read_jobs_csv",
    "validate_rows",
]
