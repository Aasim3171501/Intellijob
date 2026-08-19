"""
Integration tests for api.views.AnalyzeView.

Strategy:

* Generate a tiny real PDF in-memory using PyMuPDF (no fixture
  file shipped, no extra deps). Bytes-only, no disk touches.
* Mock pymupdf4llm.to_markdown so we control the resume text
  exactly — the test surface is the orchestration, not the PDF
  parser.
* Mock SentenceTransformer.encode via the same patch point
  matcher.py uses, so embed_query() returns a deterministic
  unit vector without a real model.
* Build a tiny npz + CSV fixture so load_job_index() has data.

The test verifies the contract: 400 on missing fields, 200 with
the right response shape, 503 when dataset is missing, and the
extracted_skills/matches/roadmap keys.
"""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from api.services.embedder import EMBEDDING_DIM
from api.services.jobs_loader import CSV_COLUMNS

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _sha_unit_vec(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_tiny_pdf_bytes() -> bytes:
    """Build a minimal valid PDF in memory using PyMuPDF. No disk I/O."""
    import pymupdf

    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "Senior Python Developer with Django and PostgreSQL experience.",
            fontsize=11,
        )
        buf = doc.tobytes()
    finally:
        doc.close()
    return buf


def _build_index_fixture(n: int = 5) -> tuple[Path, Path, list[str]]:
    """Create a tempdir with a small npz + CSV, return (tmp, csv_path, ids)."""
    tmp = Path(tempfile.mkdtemp(prefix="hermes-views-fixture-"))
    ids = [f"id-{i}" for i in range(n)]
    vectors = np.stack(
        [_sha_unit_vec(f"job-{i}") for i in range(n)],
        axis=0,
    ).astype(np.float32)
    np.savez_compressed(
        tmp / "jobs.npz",
        ids=np.asarray(ids, dtype=object),
        vectors=vectors,
        model_name=np.asarray("fake-model"),
        schema_version=np.asarray(1, dtype=np.int64),
    )
    with (tmp / "jobs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for i, rid in enumerate(ids):
            w.writerow({
                **{
                    c: "" for c in CSV_COLUMNS
                    if c not in ("id", "title", "company_display_name",
                                 "location_display", "description")
                },
                "id": rid,
                "title": f"Engineer-{i}",
                "company_display_name": f"Co-{i}",
                "location_display": f"Loc-{i}",
                "description": "Python and Django role description.",
            })
    return tmp, tmp / "jobs.csv", ids


# --------------------------------------------------------------------------- #
# Test base
# --------------------------------------------------------------------------- #


@override_settings(
    # Force matcher to read fixtures, not the real data/.
    # The view itself reads via matcher.load_job_index which is
    # already pointing at data/ — but matcher's lru_cache means the
    # FIRST call wins. We patch load_job_index to return a fixture-
    # backed JobIndex directly.
)
class AnalyzeViewTests(TestCase):
    """All view tests share a fixture-backed JobIndex built in setUpClass."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_tmp, cls.csv_path, cls.ids = _build_index_fixture(n=5)
        cls.tiny_pdf = _make_tiny_pdf_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree(cls.fixture_tmp, ignore_errors=True)

    def setUp(self) -> None:
        # Patch the JobIndex used by match_jobs so it reads our fixture.
        from api.services import matcher
        from api.services.matcher import JobIndex

        self._matcher_patch = mock.patch.object(
            matcher,
            "load_job_index",
            return_value=JobIndex(
                ids=np.asarray(self.ids, dtype=object),
                vectors=np.stack(
                    [_sha_unit_vec(f"job-{i}") for i in range(5)],
                    axis=0,
                ).astype(np.float32),
                rows={
                    rid: _fake_row(rid, i)
                    for i, rid in enumerate(self.ids)
                },
                model_name="fake",
                schema_version=1,
            ),
        )
        self._matcher_patch.start()

        # Patch the SentenceTransformer inside embed_query so no real
        # model is loaded. Returns a deterministic unit vector.
        fake_arr = np.array([_sha_unit_vec("query")])
        fake_model = mock.Mock()
        fake_model.encode = mock.Mock(return_value=fake_arr)
        self._st_patch = mock.patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_model,
        )
        self._st_patch.start()

        # Reset the query-model cache so the mock applies on first call.
        from api.services.matcher import _get_query_model
        _get_query_model.cache_clear()

        # Patch pymupdf4llm to control what "text" comes out of the PDF.
        self._pymupdf_patch = mock.patch(
            "pymupdf4llm.to_markdown",
            return_value=(
                "Senior Python developer with Django and PostgreSQL "
                "experience. AWS and Docker."
            ),
        )
        self._pymupdf_patch.start()

        # Patch the RAG roadmap service so the view never touches Ollama.
        # The service itself is covered by test_roadmap_generator.
        from api.services.roadmap_generator import SkillGapRoadmap

        self._roadmap_patch = mock.patch(
            "api.views.generate_roadmap",
            return_value=SkillGapRoadmap(
                status="generated",
                skill_gaps=["Kubernetes", "Terraform"],
                learning_steps=["Learn Kubernetes", "Learn Terraform"],
                estimated_timeline_weeks={"Phase 1 - Foundations": "2-3 weeks"},
                matched_jobs=["Engineer-0"],
                extracted_skill_count=5,
            ),
        )
        self._roadmap_mock = self._roadmap_patch.start()

        self.client = APIClient()

    def tearDown(self) -> None:
        self._roadmap_patch.stop()
        self._pymupdf_patch.stop()
        self._st_patch.stop()
        self._matcher_patch.stop()
        from api.services.matcher import _get_query_model
        _get_query_model.cache_clear()
        from api.services.matcher import load_job_index
        load_job_index.cache_clear()

    # ----------------------------------------------------------------------- #
    # Happy path
    # ----------------------------------------------------------------------- #

    def test_post_returns_expected_keys(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={
                "file": _uploaded_file(self.tiny_pdf, "resume.pdf"),
                "target_title": "Senior Backend Engineer",
            },
            format="multipart",
        )
        self.assertEqual(resp.status_code, 200, resp.content)
        body = resp.json()
        self.assertIn("extracted_skills", body)
        self.assertIn("matches", body)
        self.assertIn("roadmap", body)
        self.assertIsInstance(body["extracted_skills"], list)
        self.assertIsInstance(body["matches"], list)
        self.assertIsInstance(body["roadmap"], dict)
        # Targeted mode: title given -> no career_pathways.
        self.assertEqual(body["career_mode"], "targeted")
        self.assertEqual(body["career_pathways"], [])

    def test_post_extracts_real_skills(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={
                "file": _uploaded_file(self.tiny_pdf, "resume.pdf"),
                "target_title": "Senior Backend Engineer",
            },
            format="multipart",
        )
        body = resp.json()
        # From the mocked pymupdf4llm output.
        self.assertIn("Python", body["extracted_skills"])
        self.assertIn("Django", body["extracted_skills"])
        self.assertIn("PostgreSQL", body["extracted_skills"])
        self.assertIn("AWS", body["extracted_skills"])
        self.assertIn("Docker", body["extracted_skills"])

    def test_post_returns_top_k_matches(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={
                "file": _uploaded_file(self.tiny_pdf, "resume.pdf"),
                "target_title": "Senior Backend Engineer",
            },
            format="multipart",
        )
        body = resp.json()
        self.assertLessEqual(len(body["matches"]), 5)
        self.assertGreater(len(body["matches"]), 0)
        first = body["matches"][0]
        self.assertIn("id", first)
        self.assertIn("title", first)
        self.assertIn("company", first)
        self.assertIn("location_display", first)
        self.assertIn("similarity_score", first)
        self.assertIn("description_excerpt", first)

    def test_post_roadmap_has_schema_shape(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={
                "file": _uploaded_file(self.tiny_pdf, "resume.pdf"),
                "target_title": "Senior Backend Engineer",
            },
            format="multipart",
        )
        body = resp.json()
        roadmap = body["roadmap"]
        self.assertIsInstance(roadmap, dict)
        self.assertIn("status", roadmap)
        self.assertIn("skill_gaps", roadmap)
        self.assertIn("learning_steps", roadmap)
        self.assertIn("estimated_timeline_weeks", roadmap)
        self.assertEqual(roadmap["status"], "generated")
        self.assertIn("Kubernetes", roadmap["skill_gaps"])

    def test_post_roadmap_delegates_to_service_with_skills_and_matches(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={
                "file": _uploaded_file(self.tiny_pdf, "resume.pdf"),
                "target_title": "Senior Backend Engineer",
            },
            format="multipart",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(self._roadmap_mock.called)
        # generate_roadmap(skills, matches) — skills first, matches second.
        args = self._roadmap_mock.call_args
        self.assertEqual(len(args.args), 2)
        skills, matches = args.args
        self.assertIn("Python", skills)
        self.assertIn("Django", skills)
        self.assertIsInstance(matches, list)
        self.assertGreater(len(matches), 0)
        self.assertTrue(matches[0].title.startswith("Engineer-"))

    # ----------------------------------------------------------------------- #
    # Validation errors
    # ----------------------------------------------------------------------- #

    def test_missing_file_returns_400(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={"target_title": "Backend Engineer"},
            format="multipart",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_missing_target_title_uses_discovery_mode(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={"file": _uploaded_file(self.tiny_pdf, "r.pdf")},
            format="multipart",
        )
        # target_title is optional: blank => career discovery mode.
        self.assertEqual(resp.status_code, 200, resp.content)
        body = resp.json()
        self.assertEqual(body["career_mode"], "discovery")
        self.assertIsInstance(body["career_pathways"], list)
        self.assertGreaterEqual(len(body["career_pathways"]), 1)
        # Each pathway is self-contained.
        for pw in body["career_pathways"]:
            self.assertIn("key", pw)
            self.assertIn("name", pw)
            self.assertIn("description", pw)
            self.assertIn("similarity_score", pw)
            self.assertIn("match_count", pw)
            self.assertIn("matches", pw)
            self.assertIn("roadmap", pw)
            self.assertIsNotNone(pw["roadmap"])
        # Top-level matches/roadmap mirror the #1 pathway.
        self.assertEqual(body["matches"], body["career_pathways"][0]["matches"])
        self.assertEqual(body["roadmap"], body["career_pathways"][0]["roadmap"])

    def test_blank_target_title_uses_discovery_mode(self) -> None:
        resp = self.client.post(
            "/api/analyze/",
            data={
                "file": _uploaded_file(self.tiny_pdf, "r.pdf"),
                "target_title": "   ",
            },
            format="multipart",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["career_mode"], "discovery")

    def test_discovery_ranks_best_fit_pathway_first(self) -> None:
        """A resume whose query aligns with the 'embedded' pathway centroid
        should surface Embedded Systems as the #1 recommendation."""
        from api.services import matcher
        from api.services.pathways import build_pathway_index

        # A diverse index spanning several pathways.
        diverse = _build_diverse_index()

        self._matcher_patch.stop()
        try:
            self._matcher_patch = mock.patch.object(
                matcher,
                "load_job_index",
                return_value=diverse,
            )
            self._matcher_patch.start()

            # Align the query vector with the 'embedded' pathway centroid.
            embedded_centroid = build_pathway_index(diverse).centroids["embedded"]
            fake_arr = np.array([embedded_centroid])
            fake_model = mock.Mock()
            fake_model.encode = mock.Mock(return_value=fake_arr)
            self._st_patch.stop()
            self._st_patch = mock.patch(
                "sentence_transformers.SentenceTransformer",
                return_value=fake_model,
            )
            self._st_patch.start()
            from api.services.matcher import _get_query_model
            _get_query_model.cache_clear()

            resp = self.client.post(
                "/api/analyze/",
                data={"file": _uploaded_file(self.tiny_pdf, "r.pdf")},
                format="multipart",
            )
            self.assertEqual(resp.status_code, 200, resp.content)
            body = resp.json()
            self.assertEqual(body["career_mode"], "discovery")
            self.assertEqual(body["career_pathways"][0]["key"], "embedded")
            self.assertNotEqual(body["career_pathways"][0]["similarity_score"], 0.0)
            self.assertGreater(len(body["career_pathways"][0]["matches"]), 0)
        finally:
            self._matcher_patch = mock.patch.object(
                matcher,
                "load_job_index",
                return_value=matcher.JobIndex(
                    ids=np.asarray(self.ids, dtype=object),
                    vectors=np.stack(
                        [_sha_unit_vec(f"job-{i}") for i in range(5)],
                        axis=0,
                    ).astype(np.float32),
                    rows={
                        rid: _fake_row(rid, i)
                        for i, rid in enumerate(self.ids)
                    },
                    model_name="fake",
                    schema_version=1,
                ),
            )
            self._matcher_patch.start()
            matcher.load_job_index.cache_clear()

    def test_pdf_parser_failure_returns_400(self) -> None:
        with mock.patch(
            "pymupdf4llm.to_markdown",
            side_effect=RuntimeError("malformed PDF"),
        ):
            resp = self.client.post(
                "/api/analyze/",
                data={
                    "file": _uploaded_file(self.tiny_pdf, "r.pdf"),
                    "target_title": "Backend Engineer",
                },
                format="multipart",
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_dataset_missing_returns_503(self) -> None:
        from api.services import matcher

        # Override the patch to simulate FileNotFoundError from load.
        self._matcher_patch.stop()
        try:
            with mock.patch.object(
                matcher,
                "load_job_index",
                side_effect=FileNotFoundError(
                    "data/job_embeddings.npz missing"
                ),
            ):
                resp = self.client.post(
                    "/api/analyze/",
                    data={
                        "file": _uploaded_file(self.tiny_pdf, "r.pdf"),
                        "target_title": "Backend Engineer",
                    },
                    format="multipart",
                )
            self.assertEqual(resp.status_code, 503)
            self.assertIn("error", resp.json())
        finally:
            self._matcher_patch = mock.patch.object(
                matcher,
                "load_job_index",
                return_value=matcher.JobIndex(
                    ids=np.asarray(self.ids, dtype=object),
                    vectors=np.stack(
                        [_sha_unit_vec(f"job-{i}") for i in range(5)],
                        axis=0,
                    ).astype(np.float32),
                    rows={
                        rid: _fake_row(rid, i)
                        for i, rid in enumerate(self.ids)
                    },
                    model_name="fake",
                    schema_version=1,
                ),
            )
            self._matcher_patch.start()
            matcher.load_job_index.cache_clear()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeUpload:
    """Minimal duck-type for Django's UploadedFile.

    Django's test client wraps file-like objects in SimpleUploadedFile;
    we use that path so request.FILES.get() works.
    """


def _uploaded_file(data: bytes, name: str):
    from django.core.files.uploadedfile import SimpleUploadedFile
    return SimpleUploadedFile(name, data, content_type="application/pdf")


def _fake_row(rid: str, i: int):
    """Build a JobRow matching a fixture id without touching the CSV."""
    from api.services.jobs_loader import JobRow
    return JobRow(
        id=rid,
        title=f"Engineer-{i}",
        company_display_name=f"Co-{i}",
        location_display=f"Loc-{i}",
        location_area="",
        salary_min=None,
        salary_max=None,
        salary_is_predicted=None,
        contract_type="",
        contract_time="",
        category_label="IT Jobs",
        created="",
        description="Python and Django role description.",
        redirect_url="",
        search_query="",
        search_location="",
    )


_DIVERSE_TITLES: list[str] = [
    "Embedded Software Engineer",
    "Embedded Systems Engineer",
    "Data Scientist",
    "Data Analyst",
    "DevOps Engineer",
    "Backend Engineer",
    "Software Engineer",
    "Senior Software Engineer",
    "Frontend Developer",
]


def _build_diverse_index():
    """A JobIndex whose titles span several career pathways."""
    from api.services.jobs_loader import JobRow
    from api.services.matcher import JobIndex

    ids = [f"div-{i}" for i in range(len(_DIVERSE_TITLES))]
    vectors = np.stack(
        [_sha_unit_vec(t) for t in _DIVERSE_TITLES],
        axis=0,
    ).astype(np.float32)
    rows = {
        rid: JobRow(
            id=rid,
            title=title,
            company_display_name=f"Co-{i}",
            location_display="London",
            location_area="",
            salary_min=None,
            salary_max=None,
            salary_is_predicted=None,
            contract_type="",
            contract_time="",
            category_label="IT Jobs",
            created="",
            description="A role requiring Python and experience.",
            redirect_url="",
            search_query="",
            search_location="",
        )
        for i, (rid, title) in enumerate(zip(ids, _DIVERSE_TITLES))
    }
    return JobIndex(
        ids=np.asarray(ids, dtype=object),
        vectors=vectors,
        rows=rows,
        model_name="fake",
        schema_version=1,
    )


if __name__ == "__main__":
    unittest.main()