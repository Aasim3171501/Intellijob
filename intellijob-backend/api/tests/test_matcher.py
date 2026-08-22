"""
Tests for api.services.matcher.

Strategy:

* Build tiny npz + CSV fixtures in a tempdir (no model load, no
  real data). The vectors are sha256-seeded unit vectors so the
  ranking is reproducible.
* Mock SentenceTransformer.encode inside embed_query so the test
  runs without torch.
* Exercise match_jobs with the real npz-shaped data: shape, dtype,
  unit-norm, top-k ordering, excerpt truncation, id-join.

The "system" path (embed_query + match_jobs) is NOT tested as one
piece because that requires a real model. The intent is that the
shape + ranking logic is fully covered, and embed_query is covered
by mocking the encoder. A separate manual smoke test exercises the
end-to-end with the real model.
"""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from api.services.embedder import EMBEDDING_DIM
from api.services.jobs_loader import CSV_COLUMNS, JobRow
from api.services.matcher import (
    EXCERPT_CHARS,
    JobIndex,
    MatchResult,
    _compose_query_text,
    embed_query,
    load_job_index,
    match_from_text,
    match_jobs,
    reset_job_index_cache,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _sha_unit_vec(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic unit vector from a string seed (same idea as
    the embedder test fake)."""
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)


def _make_fake_npz(path: Path, n: int = 5) -> list[str]:
    """Write a small but well-formed npz that looks like the real one.
    Returns the ids in the order they were written."""
    ids = [f"id-{i}" for i in range(n)]
    vectors = np.stack(
        [_sha_unit_vec(f"job-{i}") for i in range(n)],
        axis=0,
    ).astype(np.float32)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=object),
        vectors=vectors,
        model_name=np.asarray("fake-model"),
        schema_version=np.asarray(1, dtype=np.int64),
    )
    return ids


def _make_fake_csv(path: Path, ids: list[str]) -> None:
    """Make a CSV with matching ids + a unique title prefix so we can
    assert ordering. description_excerpt is "desc N ..." so we can
    also check truncation."""
    rows: list[dict[str, str]] = []
    for i, rid in enumerate(ids):
        rows.append(
            {
                **{
                    c: ""
                    for c in CSV_COLUMNS
                    if c not in ("id", "title", "company_display_name",
                                 "location_display", "description")
                },
                "id": rid,
                "title": f"Engineer-{i}",
                "company_display_name": f"Co-{i}",
                "location_display": f"Loc-{i}",
                "description": "desc " * (i + 1) + " END",  # grows with i
            }
        )
    _write_csv(path, rows)


class _Fixture:
    """Bundles a tempdir + the npz + csv paths + the ids."""

    def __init__(self, n: int = 5) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hermes-matcher-"))
        self.npz = self.tmp / "jobs.npz"
        self.csv = self.tmp / "jobs.csv"
        self.ids = _make_fake_npz(self.npz, n=n)
        _make_fake_csv(self.csv, self.ids)

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# _compose_query_text
# --------------------------------------------------------------------------- #


class ComposeQueryTextTests(unittest.TestCase):
    def test_title_and_skills(self) -> None:
        out = _compose_query_text(["Python", "Django"], "Backend Engineer")
        self.assertEqual(
            out, "Target Role: Backend Engineer. Technical Skills: Python, Django"
        )

    def test_title_only(self) -> None:
        self.assertEqual(_compose_query_text([], "Data Scientist"),
                         "Target Role: Data Scientist")

    def test_skills_only(self) -> None:
        self.assertEqual(_compose_query_text(["Python"], ""),
                         "Technical Skills: Python")

    def test_empty_both(self) -> None:
        self.assertEqual(_compose_query_text([], ""), "")
        self.assertEqual(_compose_query_text(None, None), "")

    def test_skips_empty_skill_strings(self) -> None:
        out = _compose_query_text(["Python", "", "  ", "Django"], "SWE")
        self.assertEqual(out, "Target Role: SWE. Technical Skills: Python, Django")


# --------------------------------------------------------------------------- #
# embed_query (mocked encoder)
# --------------------------------------------------------------------------- #


class EmbedQueryTests(unittest.TestCase):
    def test_empty_query_returns_zero_vector(self) -> None:
        v = embed_query([], "")
        self.assertEqual(v.shape, (EMBEDDING_DIM,))
        self.assertEqual(v.dtype, np.float32)
        self.assertEqual(np.linalg.norm(v), 0.0)

    def test_returns_unit_norm(self) -> None:
        # We need to mock encode so we don't hit the real model.
        fake_arr = np.array([_sha_unit_vec("hello")])
        fake_model = mock.Mock()
        fake_model.encode = mock.Mock(return_value=fake_arr)
        with mock.patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_model,
        ):
            v = embed_query(["Python"], "Backend")
        self.assertEqual(v.shape, (EMBEDDING_DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_encoder_called_with_composed_text(self) -> None:
        fake_arr = np.array([_sha_unit_vec("x")])
        fake_model = mock.Mock()
        fake_model.encode = mock.Mock(return_value=fake_arr)
        with mock.patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_model,
        ) as fake_cls:
            embed_query(["Python", "Django"], "Backend Engineer")
        fake_cls.assert_called_once()
        fake_model.encode.assert_called_once()
        args, kwargs = fake_model.encode.call_args
        texts = args[0]
        self.assertEqual(len(texts), 1)
        self.assertIn("Backend Engineer", texts[0])
        self.assertIn("Python", texts[0])
        self.assertIn("Django", texts[0])
        self.assertTrue(kwargs.get("normalize_embeddings"))


# --------------------------------------------------------------------------- #
# load_job_index
# --------------------------------------------------------------------------- #


class LoadJobIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _Fixture(n=5)
        reset_job_index_cache()

    def tearDown(self) -> None:
        self.fx.cleanup()
        reset_job_index_cache()

    def test_loads_correct_shape_and_dtype(self) -> None:
        idx = load_job_index(
            npz_path=str(self.fx.npz),
            csv_path=str(self.fx.csv),
        )
        self.assertIsInstance(idx, JobIndex)
        self.assertEqual(idx.count, 5)
        self.assertEqual(idx.vectors.shape, (5, EMBEDDING_DIM))
        self.assertEqual(idx.vectors.dtype, np.float32)
        self.assertEqual(len(idx.ids), 5)

    def test_vectors_are_unit_norm(self) -> None:
        idx = load_job_index(
            npz_path=str(self.fx.npz),
            csv_path=str(self.fx.csv),
        )
        norms = np.linalg.norm(idx.vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)

    def test_rows_joined_by_id(self) -> None:
        idx = load_job_index(
            npz_path=str(self.fx.npz),
            csv_path=str(self.fx.csv),
        )
        self.assertEqual(len(idx.rows), 5)
        first = idx.rows[self.fx.ids[0]]
        self.assertIsInstance(first, JobRow)
        self.assertEqual(first.id, self.fx.ids[0])
        self.assertTrue(first.title.startswith("Engineer-"))

    def test_missing_csv_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_job_index(
                npz_path=str(self.fx.npz),
                csv_path=str(self.fx.tmp / "no_such_file.csv"),
            )

    def test_missing_npz_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_job_index(
                npz_path=str(self.fx.tmp / "no_such_file.npz"),
                csv_path=str(self.fx.csv),
            )


# --------------------------------------------------------------------------- #
# match_jobs (the ranking math)
# --------------------------------------------------------------------------- #


class MatchJobsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _Fixture(n=5)
        reset_job_index_cache()
        self.idx = load_job_index(
            npz_path=str(self.fx.npz),
            csv_path=str(self.fx.csv),
        )

    def tearDown(self) -> None:
        self.fx.cleanup()
        reset_job_index_cache()

    def test_returns_top_k_in_similarity_order(self) -> None:
        # Query = one of the indexed job vectors verbatim => top match should be that job.
        q = self.idx.vectors[2].copy()
        results = match_jobs(q, top_k=3, index=self.idx)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].id, self.fx.ids[2])
        # Similarities should be monotonically non-increasing.
        sims = [r.similarity for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_top_k_caps_at_index_size(self) -> None:
        q = self.idx.vectors[0].copy()
        results = match_jobs(q, top_k=99, index=self.idx)
        self.assertEqual(len(results), 5)

    def test_top_k_zero_returns_empty(self) -> None:
        results = match_jobs(self.idx.vectors[0], top_k=0, index=self.idx)
        self.assertEqual(results, [])

    def test_similarity_in_range_for_unit_vectors(self) -> None:
        q = self.idx.vectors[0].copy()
        results = match_jobs(q, top_k=5, index=self.idx)
        for r in results:
            self.assertGreaterEqual(r.similarity, -1.0)
            self.assertLessEqual(r.similarity, 1.0)

    def test_query_normalisation_when_unnormalised(self) -> None:
        # Pass a deliberately unnormalised vector (5x the unit).
        q = self.idx.vectors[0] * 5.0
        results_unit = match_jobs(
            self.idx.vectors[0].copy(), top_k=1, index=self.idx,
        )
        results_scaled = match_jobs(q, top_k=1, index=self.idx)
        # Same top-1, similarity must match within float tolerance.
        self.assertEqual(results_unit[0].id, results_scaled[0].id)
        self.assertAlmostEqual(
            results_unit[0].similarity,
            results_scaled[0].similarity,
            places=4,
        )

    def test_invalid_query_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            match_jobs(np.zeros(EMBEDDING_DIM + 1, dtype=np.float32),
                       index=self.idx)

    def test_match_result_has_all_fields(self) -> None:
        q = self.idx.vectors[1].copy()
        r = match_jobs(q, top_k=1, index=self.idx)[0]
        self.assertIsInstance(r, MatchResult)
        self.assertEqual(r.id, self.fx.ids[1])
        self.assertTrue(r.title.startswith("Engineer-"))
        self.assertTrue(r.company_display_name.startswith("Co-"))
        self.assertTrue(r.location_display.startswith("Loc-"))
        self.assertIn("desc", r.description_excerpt.lower())

    def test_description_excerpt_is_truncated(self) -> None:
        # Build a fixture where one row's description exceeds EXCERPT_CHARS.
        big = "X" * (EXCERPT_CHARS + 50)
        # Replace fixture CSV's first row description with the big one.
        # Easier: build a focused 2-row fixture.
        tmp = Path(tempfile.mkdtemp(prefix="hermes-matcher-trunc-"))
        try:
            ids = ["a", "b"]
            np.savez_compressed(
                tmp / "x.npz",
                ids=np.asarray(ids, dtype=object),
                vectors=np.stack(
                    [_sha_unit_vec("a"), _sha_unit_vec("b")],
                    axis=0,
                ).astype(np.float32),
                model_name=np.asarray("m"),
                schema_version=np.asarray(1, dtype=np.int64),
            )
            _write_csv(tmp / "x.csv", [
                {**{c: "" for c in CSV_COLUMNS}, 
                    "id": "a", "title": "A",
                    "description": big
                },
                {**{c: "" for c in CSV_COLUMNS}, 
                    "id": "b", "title": "B",
                    "description": "short"
                },
            ])
            reset_job_index_cache()
            idx = load_job_index(
                npz_path=str(tmp / "x.npz"),
                csv_path=str(tmp / "x.csv"),
            )
            r = match_jobs(idx.vectors[0].copy(), top_k=1, index=idx)[0]
            # Should be truncated with an ellipsis character.
            self.assertLess(len(r.description_excerpt), len(big))
            self.assertTrue(r.description_excerpt.endswith("…"))
            # Within budget + ellipsis.
            self.assertLessEqual(len(r.description_excerpt), EXCERPT_CHARS + 5)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            reset_job_index_cache()

    def test_to_dict_round_trip(self) -> None:
        q = self.idx.vectors[0].copy()
        r = match_jobs(q, top_k=1, index=self.idx)[0]
        d = r.to_dict()
        self.assertEqual(d["id"], r.id)
        self.assertEqual(d["title"], r.title)
        self.assertEqual(d["company"], r.company_display_name)
        self.assertEqual(d["location_display"], r.location_display)
        self.assertEqual(d["similarity_score"], round(r.similarity, 6))
        self.assertEqual(d["description_excerpt"], r.description_excerpt)

    def test_missing_id_in_csv_is_skipped(self) -> None:
        # Build an index whose npz has an id that does NOT appear in the CSV.
        tmp = Path(tempfile.mkdtemp(prefix="hermes-matcher-missing-"))
        try:
            ids = ["a", "ghost"]
            np.savez_compressed(
                tmp / "x.npz",
                ids=np.asarray(ids, dtype=object),
                vectors=np.stack(
                    [_sha_unit_vec("a"), _sha_unit_vec("ghost")],
                    axis=0,
                ).astype(np.float32),
                model_name=np.asarray("m"),
                schema_version=np.asarray(1, dtype=np.int64),
            )
            # CSV only contains "a".
            _write_csv(tmp / "x.csv", [
                {**{c: "" for c in CSV_COLUMNS}, 
                    "id": "a", "title": "A",
                    "description": "the only row"
                },
            ])
            reset_job_index_cache()
            idx = load_job_index(
                npz_path=str(tmp / "x.npz"),
                csv_path=str(tmp / "x.csv"),
            )
            results = match_jobs(
                idx.vectors[0].copy(), top_k=2, index=idx,
            )
            # The ghost row should be silently skipped; we get 1 result.
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].id, "a")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            reset_job_index_cache()


# --------------------------------------------------------------------------- #
# match_from_text (the convenience wrapper — encoder mocked)
# --------------------------------------------------------------------------- #


class MatchFromTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _Fixture(n=5)
        reset_job_index_cache()

    def tearDown(self) -> None:
        self.fx.cleanup()
        reset_job_index_cache()

    def test_returns_matches(self) -> None:
        # Force a deterministic query vector via the fake encoder.
        fake_query = _sha_unit_vec("python backend")
        fake_arr = np.array([fake_query])
        fake_model = mock.Mock()
        fake_model.encode = mock.Mock(return_value=fake_arr)

        # Patch at the matcher module's import point (embed_query uses
        # sentence_transformers directly).
        with mock.patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_model,
        ):
            results = match_from_text(
                ["Python"], "Backend Engineer", top_k=3,
            )
        self.assertEqual(len(results), 3)
        # All results should have valid similarity scores.
        for r in results:
            self.assertIsInstance(r.similarity, float)
            self.assertGreaterEqual(r.similarity, -1.0)
            self.assertLessEqual(r.similarity, 1.0)


if __name__ == "__main__":
    unittest.main()