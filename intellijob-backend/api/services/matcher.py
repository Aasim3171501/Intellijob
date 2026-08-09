"""
IntelliJob — Phase B-2: cosine-similarity matching against the
pre-vectorised job dataset.

This module loads the .npz produced by :mod:`scripts.embed_jobs`
and the CSV produced by :mod:`scripts.seed_jobs`, and provides:

    embed_query(skills, target_title) -> np.ndarray (D,)
    match_jobs(query_vec, top_k) -> list[MatchResult]
    match_from_text(skills, target_title, top_k) -> list[MatchResult]

Design:

* The npz and CSV are loaded lazily, once per process, and cached
  on module scope. The data is small (~1.4 MB vectors + 858 KB
  CSV) so paying the cost on first request is fine.

* All vectors in the npz are unit-norm (the embedder normalises
  them), so cosine distance == negative dot product. For ranking
  top-k we use ``argpartition`` (O(N) rather than O(N log N)) on
  the matrix-vector product ``V @ q``.

* The query vector is built by composing a short text — the target
  title plus the extracted skills — and re-embedding with the same
  SentenceTransformer that produced the job vectors. That keeps the
  user-side query in the same vector space as the dataset.

* MatchResult is a frozen dataclass with the join keys the API
  view needs (id, title, company, location_display, similarity,
  description_excerpt). The CSV is the source of truth for the
  human-readable fields, the npz for the vectors.

Public API:
    load_job_index() -> JobIndex
    embed_query(skills, target_title) -> np.ndarray
    match_jobs(query_vec, top_k) -> list[MatchResult]
    match_from_text(skills, target_title, top_k) -> list[MatchResult]
    MatchResult, JobIndex
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from api.services.embedder import DEFAULT_MODEL_NAME, EMBEDDING_DIM
from api.services.jobs_loader import JobRow, read_jobs_csv

# --------------------------------------------------------------------------- #
# Defaults — match the Phase B-1 manifest schema.
# --------------------------------------------------------------------------- #

_DEFAULT_NPZ = Path(__file__).resolve().parents[2] / "data" / "job_embeddings.npz"
_DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "uk_software_jobs.csv"

#: Excerpt length for description_preview in MatchResult.
EXCERPT_CHARS: int = 240


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MatchResult:
    """One job spec ranked against the candidate query.

    Fields are shaped for direct JSON serialisation by the view
    layer. ``similarity`` is in [0, 1] for unit-norm vectors
    (dot product == cosine).
    """

    id: str
    title: str
    company_display_name: str
    location_display: str
    similarity: float
    description_excerpt: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company_display_name,
            "location_display": self.location_display,
            "similarity_score": round(float(self.similarity), 6),
            "description_excerpt": self.description_excerpt,
        }


# --------------------------------------------------------------------------- #
# Index loader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobIndex:
    """In-memory view of the pre-vectorised job dataset.

    ``ids`` and ``vectors`` are parallel: ``vectors[i]`` is the
    embedding for ``ids[i]``. ``rows`` is keyed by job id and
    provides the human-readable fields for the response.
    """

    ids: np.ndarray           # shape (N,), dtype object
    vectors: np.ndarray       # shape (N, 384), dtype float32, unit-norm
    rows: dict[str, JobRow]   # id -> JobRow (for join)
    model_name: str
    schema_version: int

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])


@lru_cache(maxsize=1)
def load_job_index(
    npz_path: str | None = None,
    csv_path: str | None = None,
) -> JobIndex:
    """Load the .npz + CSV pair into a JobIndex. Cached at module
    scope — repeated calls return the same object. Pass alternate
    paths in tests; production callers use the defaults.
    """
    npz_p = Path(npz_path) if npz_path else _DEFAULT_NPZ
    csv_p = Path(csv_path) if csv_path else _DEFAULT_CSV

    if not npz_p.exists():
        raise FileNotFoundError(
            f"Job embeddings npz not found: {npz_p}. "
            f"Run scripts/embed_jobs.py first."
        )
    if not csv_p.exists():
        raise FileNotFoundError(f"Job CSV not found: {csv_p}")

    with np.load(npz_p, allow_pickle=True) as raw:
        ids = raw["ids"]
        vectors = raw["vectors"].astype(np.float32, copy=False)
        model_name = str(raw["model_name"].item()) if raw["model_name"].ndim == 0 else str(raw["model_name"])
        schema_version = int(raw["schema_version"])

    rows: dict[str, JobRow] = {str(r.id): r for r in read_jobs_csv(csv_p)}

    return JobIndex(
        ids=ids,
        vectors=vectors,
        rows=rows,
        model_name=model_name,
        schema_version=schema_version,
    )


def reset_job_index_cache() -> None:
    """Clear the lru_cache — used by tests after writing a new npz."""
    load_job_index.cache_clear()


# --------------------------------------------------------------------------- #
# Query embedding
# --------------------------------------------------------------------------- #


def _compose_query_text(skills: Sequence[str], target_title: str) -> str:
    """The single source of truth for query composition.

    Composition: "{target_title}. skills: skill1, skill2, ..."

    Stays next to :func:`embed_query` for easy evolution — if this
    ever diverges from the dataset's embedder-side composition,
    top-k ranking degrades silently. Bump the matcher schema_version
    if you change it.
    """
    title = (target_title or "").strip()
    skills_clean = [s for s in (skills or []) if s and s.strip()]
    parts: list[str] = []
    if title:
        parts.append(title)
    if skills_clean:
        parts.append("skills: " + ", ".join(skills_clean))
    return ". ".join(parts)


@lru_cache(maxsize=1)
def _get_query_model(model_name: str):
    """Cached SentenceTransformer for query embedding. Built lazily
    on first call so module-import stays free of torch. Tests can
    call ``_get_query_model.cache_clear()`` to force a rebuild."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name, device="cpu")
    model.eval()
    return model


def embed_query(
    skills: Sequence[str],
    target_title: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> np.ndarray:
    """Embed a candidate query (skills + target title) into a 384-d
    unit-norm vector, in the same space as the pre-vectorised jobs.

    Returns the all-zero vector (length 384) for an entirely empty
    query — matches are then all-zero similarity, a stable but
    uninformative fallback. Callers should reject empty input.

    The SentenceTransformer is loaded lazily and cached at module
    scope (see :func:`_get_query_model`). First call pays the
    ~10 s model-load cost; subsequent calls are sub-second on CPU.
    """
    text = _compose_query_text(skills, target_title)
    if not text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    model = _get_query_model(model_name)
    vec = model.encode(
        [text],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0].astype(np.float32, copy=False)
    return vec


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def match_jobs(
    query_vec: np.ndarray,
    *,
    top_k: int = 5,
    index: JobIndex | None = None,
) -> list[MatchResult]:
    """Rank every job in the index by cosine similarity to
    ``query_vec`` and return the top-k as MatchResults.

    Vectors in the index are unit-norm; we normalise the query
    defensively too in case the embedder returned unnormalised
    output (e.g. if normalisation is ever disabled).
    """
    if index is None:
        index = load_job_index()

    if query_vec.ndim != 1:
        raise ValueError(
            f"query_vec must be 1-D (D,); got shape {query_vec.shape!r}"
        )
    if query_vec.shape[0] != index.vectors.shape[1]:
        raise ValueError(
            f"query_vec dim ({query_vec.shape[0]}) does not match "
            f"index dim ({index.vectors.shape[1]})"
        )
    if top_k <= 0:
        return []

    # Defensive normalisation. np.linalg.norm is the safe one-liner.
    q = query_vec.astype(np.float32, copy=False)
    qn = float(np.linalg.norm(q))
    if qn > 0:
        q = q / qn

    # scores shape (N,)
    scores = index.vectors @ q

    # argpartition is O(N); argsort would be O(N log N) and we only
    # need the top-k. We then sort those k by score descending.
    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    out: list[MatchResult] = []
    for i in top_idx:
        job_id = str(index.ids[i])
        score = float(scores[i])
        row = index.rows.get(job_id)
        if row is None:
            # Defensive — npz id not in CSV. Skip rather than 500.
            continue
        excerpt = (row.description or "").strip()
        if len(excerpt) > EXCERPT_CHARS:
            excerpt = excerpt[:EXCERPT_CHARS].rstrip() + "…"
        out.append(
            MatchResult(
                id=job_id,
                title=row.title,
                company_display_name=row.company_display_name,
                location_display=row.location_display,
                similarity=score,
                description_excerpt=excerpt,
            )
        )
    return out


def match_from_text(
    skills: Sequence[str],
    target_title: str,
    *,
    top_k: int = 5,
) -> list[MatchResult]:
    """Convenience: embed the query and rank in one call."""
    q = embed_query(skills, target_title)
    return match_jobs(q, top_k=top_k)


__all__ = [
    "EXCERPT_CHARS",
    "JobIndex",
    "MatchResult",
    "load_job_index",
    "reset_job_index_cache",
    "embed_query",
    "match_jobs",
    "match_from_text",
]