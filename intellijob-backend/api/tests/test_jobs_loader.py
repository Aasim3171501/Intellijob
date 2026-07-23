"""
Smoke tests for api.services.jobs_loader.

These tests do NOT require Django, the database, or any external
service — they build a small CSV in a tempdir and exercise the
parser + validator. That makes them safe to run as the first
``python manage.py test`` check.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from api.services.jobs_loader import (
    CSV_COLUMNS,
    MIN_DESCRIPTION_CHARS,
    JobRow,
    read_jobs_csv,
    validate_rows,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _good_description(n: int = MIN_DESCRIPTION_CHARS + 50) -> str:
    return "x" * n


class ReadJobsCsvTests(unittest.TestCase):
    def test_returns_typed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "jobs.csv"
            _write_csv(
                p,
                [
                    {
                        "id": "abc123",
                        "title": "Software Engineer",
                        "company_display_name": "Acme",
                        "location_display": "London",
                        "location_area": "UK, London",
                        "salary_min": "50000",
                        "salary_max": "70000",
                        "salary_is_predicted": "1",
                        "contract_type": "permanent",
                        "contract_time": "full_time",
                        "category_label": "IT Jobs",
                        "created": "2025-01-01T00:00:00Z",
                        "description": _good_description(),
                        "redirect_url": "https://example.com/jobs/abc123",
                        "search_query": "software engineer",
                        "search_location": "London",
                    }
                ],
            )
            rows = read_jobs_csv(p)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertIsInstance(r, JobRow)
        self.assertEqual(r.id, "abc123")
        self.assertEqual(r.title, "Software Engineer")
        self.assertEqual(r.salary_min, 50000)
        self.assertEqual(r.salary_max, 70000)
        self.assertEqual(r.salary_is_predicted, True)

    def test_missing_salary_becomes_none_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "jobs.csv"
            _write_csv(
                p,
                [
                    {
                        "id": "x",
                        "title": "T",
                        "company_display_name": "",
                        "location_display": "",
                        "location_area": "",
                        "salary_min": "",
                        "salary_max": "",
                        "salary_is_predicted": "",
                        "contract_type": "",
                        "contract_time": "",
                        "category_label": "",
                        "created": "",
                        "description": _good_description(),
                        "redirect_url": "",
                        "search_query": "",
                        "search_location": "",
                    }
                ],
            )
            rows = read_jobs_csv(p)
        self.assertIsNone(rows[0].salary_min)
        self.assertIsNone(rows[0].salary_max)
        self.assertIsNone(rows[0].salary_is_predicted)

    def test_salary_float_string_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "jobs.csv"
            _write_csv(
                p,
                [
                    {
                        "id": "x",
                        "title": "T",
                        "company_display_name": "",
                        "location_display": "",
                        "location_area": "",
                        "salary_min": "45000.0",
                        "salary_max": "65000.0",
                        "salary_is_predicted": "true",
                        "contract_type": "",
                        "contract_time": "",
                        "category_label": "",
                        "created": "",
                        "description": _good_description(),
                        "redirect_url": "",
                        "search_query": "",
                        "search_location": "",
                    }
                ],
            )
            rows = read_jobs_csv(p)
        self.assertEqual(rows[0].salary_min, 45000)
        self.assertEqual(rows[0].salary_max, 65000)
        self.assertEqual(rows[0].salary_is_predicted, True)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_jobs_csv("/no/such/path/does/not/exist.csv")

    def test_out_of_order_columns_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "jobs.csv"
            # Hand-craft a CSV whose header order differs from CSV_COLUMNS.
            p.write_text(
                "title,id,description\n"
                "Engineer,xyz,hello\n",
                encoding="utf-8",
            )
            rows = read_jobs_csv(p)
        self.assertEqual(rows[0].id, "xyz")
        self.assertEqual(rows[0].title, "Engineer")
        self.assertEqual(rows[0].description, "hello")


class ValidateRowsTests(unittest.TestCase):
    def test_clean_dataset_has_no_flags(self) -> None:
        rows = [
            JobRow(
                id="1", title="SWE", company_display_name="A",
                location_display="London", location_area="UK",
                salary_min=None, salary_max=None, salary_is_predicted=None,
                contract_type="", contract_time="", category_label="",
                created="", description=_good_description(),
                redirect_url="", search_query="", search_location="",
            )
        ]
        rep = validate_rows(rows)
        self.assertEqual(rep.total, 1)
        self.assertEqual(rep.clean_count, 1)
        self.assertEqual(rep.flagged_indices, [])

    def test_flags_missing_id_title_description(self) -> None:
        rows = [
            JobRow(
                id="", title="", company_display_name="",
                location_display="", location_area="",
                salary_min=None, salary_max=None, salary_is_predicted=None,
                contract_type="", contract_time="", category_label="",
                created="", description="",
                redirect_url="", search_query="", search_location="",
            )
        ]
        rep = validate_rows(rows)
        self.assertEqual(rep.missing_id, 1)
        self.assertEqual(rep.missing_title, 1)
        self.assertEqual(rep.missing_description, 1)
        self.assertIn(0, rep.flagged_indices)

    def test_flags_short_description_only(self) -> None:
        rows = [
            JobRow(
                id="1", title="SWE", company_display_name="",
                location_display="London", location_area="",
                salary_min=None, salary_max=None, salary_is_predicted=None,
                contract_type="", contract_time="", category_label="",
                created="", description="too short",
                redirect_url="", search_query="", search_location="",
            )
        ]
        rep = validate_rows(rows)
        self.assertEqual(rep.short_description, 1)
        self.assertEqual(rep.missing_description, 0)
        self.assertIn(0, rep.flagged_indices)

    def test_flags_duplicate_id_only_on_second_occurrence(self) -> None:
        rows = [
            JobRow(
                id="dup", title="SWE", company_display_name="",
                location_display="London", location_area="",
                salary_min=None, salary_max=None, salary_is_predicted=None,
                contract_type="", contract_time="", category_label="",
                created="", description=_good_description(),
                redirect_url="", search_query="", search_location="",
            ),
            JobRow(
                id="dup", title="SWE", company_display_name="",
                location_display="London", location_area="",
                salary_min=None, salary_max=None, salary_is_predicted=None,
                contract_type="", contract_time="", category_label="",
                created="", description=_good_description(),
                redirect_url="", search_query="", search_location="",
            ),
        ]
        rep = validate_rows(rows)
        self.assertEqual(rep.duplicate_id, 1)
        self.assertEqual(rep.clean_count, 1)
        self.assertEqual(rep.flagged_indices, [1])

    def test_require_uk_flags_non_uk_locations(self) -> None:
        rows = [
            JobRow(
                id="1", title="SWE", company_display_name="",
                location_display="London", location_area="",
                salary_min=None, salary_max=None, salary_is_predicted=None,
                contract_type="", contract_time="", category_label="",
                created="", description=_good_description(),
                redirect_url="", search_query="", search_location="",
            ),
            JobRow(
                id="2", title="SWE", company_display_name="",
                location_display="Berlin", location_area="",
                salary_min=None, salary_max=None, salary_is_predicted=None,
                contract_type="", contract_time="", category_label="",
                created="", description=_good_description(),
                redirect_url="", search_query="", search_location="",
            ),
        ]
        rep_no_uk = validate_rows(rows, require_uk=False)
        self.assertEqual(rep_no_uk.non_uk_location, 0)
        rep_uk = validate_rows(rows, require_uk=True)
        self.assertEqual(rep_uk.non_uk_location, 1)
        self.assertIn(1, rep_uk.flagged_indices)

    def test_summary_is_one_line(self) -> None:
        rep = validate_rows([])
        s = rep.summary()
        self.assertIn("total=0", s)
        self.assertIn("clean=0", s)
