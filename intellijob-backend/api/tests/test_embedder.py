"""
Smoke tests for api.services.embedder.

These tests do NOT require torch or sentence-transformers to be
installed — they inject a fake SentenceTransformer into the module's
``_load_model`` helper via ``unittest.mock``. Tests that exercise the
real model would need the ~500MB install + model download, and they
belong in a separate, opt-in integration test (not here).

What's covered:

  * compose_embedding_text — field inclusion / exclusion / ordering.
  * EmbeddingResult — shape, dtype, dim, length checks.
  * embed_jobs — empty input short-circuit, deterministic with a fake
    deterministic encoder, parallel id/vector layout.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from api.services.embedder import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDING_SCHEMA_VERSION,
    EmbeddingResult,
    compose_embedding_text,
    embed_jobs,
)
from api.services.jobs_loader import JobRow


def _row(
    *,
    id: str = "1",
    title: str = "Software Engineer",
    description: str = "Python and Django",
    category_label: str = "IT Jobs",
    company_display_name: str = "Acme",
    location_display: str = "London",
    location_area: str = "UK",
) -> JobRow:
    return JobRow(
        id=id,
        title=title,
        company_display_name=company_display_name,
        location_display=location_display,
        location_area=location_area,
        salary_min=None,
        salary_max=None,
        salary_is_predicted=None,
        contract_type="",
        contract_time="",
        category_label=category_label,
        created="",
        description=description,
        redirect_url="",
        search_query="",
        search_location="",
    )


class ComposeEmbeddingTextTests(unittest.TestCase):
    def test_contains_title_category_description_in_order(self) -> None:
        r = _row(title="Backend Engineer", category_label="IT Jobs",
                 description="Python and Django required.")
        out = compose_embedding_text(r)
        self.assertTrue(out.startswith("Backend Engineer. IT Jobs."))
        self.assertIn("Python and Django required.", out)

    def test_omits_empty_fields_without_dangling_separator(self) -> None:
        r = _row(title="Data Scientist", category_label="",
                 description="Pandas and scikit-learn")
        out = compose_embedding_text(r)
        self.assertEqual(out, "Data Scientist. Pandas and scikit-learn")

    def test_description_only(self) -> None:
        r = _row(title="", category_label="", description="Python role")
        out = compose_embedding_text(r)
        self.assertEqual(out, "Python role")

    def test_empty_row_returns_empty_string(self) -> None:
        r = _row(title="", category_label="", description="")
        out = compose_embedding_text(r)
        self.assertEqual(out, "")

    def test_does_not_leak_company_location_salary(self) -> None:
        r = _row(
            title="SWE",
            description="Python",
            company_display_name="BigCorp",
            location_display="Manchester",
            location_area="UK, Manchester",
        )
        out = compose_embedding_text(r)
        self.assertNotIn("BigCorp", out)
        self.assertNotIn("Manchester", out)
        self.assertNotIn("UK", out)


class EmbeddingResultTests(unittest.TestCase):
    def _good(self) -> EmbeddingResult:
        return EmbeddingResult(
            ids=["a", "b"],
            vectors=np.zeros((2, EMBEDDING_DIM), dtype=np.float32),
            model_name=DEFAULT_MODEL_NAME,
            schema_version=EMBEDDING_SCHEMA_VERSION,
        )

    def test_construction_accepts_valid_payload(self) -> None:
        e = self._good()
        self.assertEqual(e.count, 2)
        self.assertEqual(e.ids, ["a", "b"])

    def test_rejects_wrong_dim(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingResult(
                ids=["a"],
                vectors=np.zeros((1, 256), dtype=np.float32),
                model_name=DEFAULT_MODEL_NAME,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            )

    def test_rejects_wrong_dtype(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingResult(
                ids=["a"],
                vectors=np.zeros((1, EMBEDDING_DIM), dtype=np.float64),
                model_name=DEFAULT_MODEL_NAME,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            )

    def test_rejects_mismatched_length(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingResult(
                ids=["a", "b", "c"],
                vectors=np.zeros((2, EMBEDDING_DIM), dtype=np.float32),
                model_name=DEFAULT_MODEL_NAME,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            )

    def test_rejects_1d_vectors(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingResult(
                ids=["a"],
                vectors=np.zeros((EMBEDDING_DIM,), dtype=np.float32),
                model_name=DEFAULT_MODEL_NAME,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            )


class EmbedJobsTests(unittest.TestCase):
    """All tests in this class inject a fake model so torch is never
    imported. The fake encoder just hashes each text deterministically
    into a 384-d unit vector — enough to assert shape, dtype, parallel
    ids, and seed-determinism."""

    @staticmethod
    def _fake_encoder(
        texts: list[str],
        dim: int = EMBEDDING_DIM,
        **_unused_kwargs: object,
    ) -> np.ndarray:
        """Stand-in for SentenceTransformer.encode. Accepts and
        ignores any kwargs (batch_size, show_progress_bar, ...) the
        real encoder would also accept."""
        import hashlib

        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed_bytes = hashlib.sha256(t.encode("utf-8")).digest()
            # Use the first 8 bytes as a uint64 seed for reproducibility.
            rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
            vec = rng.standard_normal(out.shape[1]).astype(out.dtype, copy=False)
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
            out[i] = vec
        return out

    def test_empty_input_short_circuits(self) -> None:
        # No model load happens for empty input.
        with mock.patch(
            "api.services.embedder._load_model"
        ) as fake_load:
            res = embed_jobs([])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.ids, [])
        self.assertEqual(res.vectors.shape, (0, EMBEDDING_DIM))
        self.assertEqual(res.vectors.dtype, np.float32)
        fake_load.assert_not_called()

    def test_returns_correct_shape_and_parallel_ids(self) -> None:
        rows = [_row(id=str(i), title=f"role {i}") for i in range(5)]
        with mock.patch(
            "api.services.embedder._load_model",
            return_value=mock.Mock(encode=self._fake_encoder),
        ):
            res = embed_jobs(rows, seed=42)
        self.assertEqual(res.count, 5)
        self.assertEqual(res.vectors.shape, (5, EMBEDDING_DIM))
        self.assertEqual(res.vectors.dtype, np.float32)
        self.assertEqual(res.ids, ["0", "1", "2", "3", "4"])
        self.assertEqual(res.schema_version, EMBEDDING_SCHEMA_VERSION)
        self.assertEqual(res.model_name, DEFAULT_MODEL_NAME)

    def test_embeddings_are_unit_norm(self) -> None:
        # The fake encoder returns unit vectors; embed_jobs reuses them,
        # so we can assert row norms are 1.
        rows = [_row(id=str(i), title=f"role {i}") for i in range(3)]
        with mock.patch(
            "api.services.embedder._load_model",
            return_value=mock.Mock(encode=self._fake_encoder),
        ):
            res = embed_jobs(rows)
        norms = np.linalg.norm(res.vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_deterministic_with_same_seed(self) -> None:
        rows = [_row(id=str(i), description=f"desc {i}") for i in range(4)]
        with mock.patch(
            "api.services.embedder._load_model",
            return_value=mock.Mock(encode=self._fake_encoder),
        ):
            a = embed_jobs(rows, seed=42)
        with mock.patch(
            "api.services.embedder._load_model",
            return_value=mock.Mock(encode=self._fake_encoder),
        ):
            b = embed_jobs(rows, seed=42)
        np.testing.assert_array_equal(a.vectors, b.vectors)
        self.assertEqual(a.ids, b.ids)

    def test_seeding_changes_python_state(self) -> None:
        # Different seed -> different _seed_everything effect via NumPy;
        # since our fake encoder is text-deterministic, vector equality
        # across texts is identical, but we still assert that calling
        # with different seeds does not raise and produces same shape.
        rows = [_row(id="x"), _row(id="y", title="diff")]
        with mock.patch(
            "api.services.embedder._load_model",
            return_value=mock.Mock(encode=self._fake_encoder),
        ):
            a = embed_jobs(rows, seed=1)
            b = embed_jobs(rows, seed=2)
        self.assertEqual(a.vectors.shape, b.vectors.shape)
        self.assertEqual(a.ids, b.ids)

    def test_passes_through_model_name(self) -> None:
        rows = [_row()]
        fake_model = mock.Mock(encode=self._fake_encoder)
        with mock.patch(
            "api.services.embedder._load_model",
            return_value=fake_model,
        ) as fake_load:
            embed_jobs(rows, model_name="custom/model", device="cpu")
        fake_load.assert_called_once_with(
            model_name="custom/model", device="cpu"
        )


if __name__ == "__main__":
    unittest.main()
