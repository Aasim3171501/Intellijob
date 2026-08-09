"""
IntelliJob — Phase B-1: text embedding for jobs (and later, resumes).

This module turns a list of :class:`JobRow` into a NumPy array of
dense vectors using a SentenceTransformers model. The default model is
``all-MiniLM-L6-v2`` (384-dim), which is the one named in
``Specifications.txt`` and ``README.md`` — so it must NOT be changed
without updating those documents.

Design notes:

* Pure Python + NumPy. No Django, no pgvector, no DB. This keeps the
  embedder unit-testable in milliseconds and lets us defer the
  Postgres + pgvector storage layer to a later session without
  rewriting any of this code.

* The interface is intentionally minimal: pass a list of JobRow, get
  back a NumPy array of shape ``(N, 384)`` plus parallel arrays of
  ids. Everything else (model loading, device choice, batching,
  seeding) is configuration.

* Determinism: SentenceTransformers in PyTorch is not bit-for-bit
  deterministic across non-CUDA runs by default, but we set the
  random seed before encoding AND we ``model.eval()`` so inference
  behaves consistently. We do NOT claim exact reproducibility
  across CPU/GPU/macOS/Windows — only "same code, same machine,
  same seed = same output".

* The composition of the embedding text is part of the public API
  (see :func:`compose_embedding_text`). Changing it changes the
  vector space and would invalidate any precomputed index — bump
  ``EMBEDDING_SCHEMA_VERSION`` when you change it.

Public API:
    EMBEDDING_SCHEMA_VERSION   - int, bump when embedding_text changes
    DEFAULT_MODEL_NAME         - str, all-MiniLM-L6-v2 per spec
    EMBEDDING_DIM              - int, 384 for the default model
    compose_embedding_text(row) - compose the text fed to the model
    embed_jobs(rows, ...)      - encode a list of JobRow into vectors
    EmbeddingResult            - dataclass, vectors + parallel ids
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from api.services.jobs_loader import JobRow

# --------------------------------------------------------------------------- #
# Schema / model constants
# --------------------------------------------------------------------------- #

#: Bump this when compose_embedding_text() changes shape or ordering.
#: Downstream consumers (loaders, indexes) should refuse to use a file
#: whose schema_version does not match.
EMBEDDING_SCHEMA_VERSION: int = 1

#: The model named in the dissertation specification. Do not change
#: without updating Specifications.txt, README.md, and the Pyproject.
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Output dimensionality of the default model. Other MiniLM variants
#: have the same dim, but larger models do not — EMBEDDING_DIM is the
#: single source of truth that downstream pgvector columns must use.
EMBEDDING_DIM: int = 384


# --------------------------------------------------------------------------- #
# Embedding text composition
# --------------------------------------------------------------------------- #


def compose_embedding_text(row: JobRow) -> str:
    """Build the single string that gets fed to the embedding model.

    Order and shape are part of the public contract — see
    ``EMBEDDING_SCHEMA_VERSION``.

    Composition (intentionally simple — the spec says the embedding
    must be "deterministic" and "context-grounded"):

        "{title}. {category_label}. {description}"

    Rationale for each field:
      * title — strongest single signal of role.
      * category_label — adds role-level noise but is free and stable.
      * description — bulk of the skill signal; HTML-stripped by the
        seeder already.

    Fields deliberately NOT included:
      * company_display_name — noise.
      * location_display / location_area — would introduce geographic
        bias that we don't want for skill matching.
      * salary_* — irrelevant to skill similarity.
      * contract_type / contract_time — too generic; not worth the
        embedding-noise cost.
    """
    parts: list[str] = []
    title = (row.title or "").strip()
    if title:
        parts.append(title)
    category = (row.category_label or "").strip()
    if category:
        parts.append(category)
    description = (row.description or "").strip()
    if description:
        parts.append(description)
    return ". ".join(parts)


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EmbeddingResult:
    """The output of :func:`embed_jobs`.

    ``ids`` and ``vectors`` are parallel: ``vectors[i]`` is the
    embedding for ``ids[i]``. Empty or missing ids raise at construction
    time so we never silently lose the join key.
    """

    ids: list[str]
    vectors: np.ndarray  # shape (N, EMBEDDING_DIM), dtype float32
    model_name: str
    schema_version: int

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (N, dim); got shape {self.vectors.shape!r}"
            )
        if self.vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"vectors dim must be {EMBEDDING_DIM}; got {self.vectors.shape[1]}"
            )
        if self.vectors.dtype != np.float32:
            raise ValueError(
                f"vectors dtype must be float32; got {self.vectors.dtype}"
            )
        if len(self.ids) != self.vectors.shape[0]:
            raise ValueError(
                f"ids length ({len(self.ids)}) must match vectors rows "
                f"({self.vectors.shape[0]})"
            )

    @property
    def count(self) -> int:
        return self.vectors.shape[0]


# --------------------------------------------------------------------------- #
# Core encoder
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    """Best-effort determinism across Python, NumPy, and (if loaded) torch.

    SentenceTransformers wraps a PyTorch model. We set the Python and
    NumPy seeds unconditionally, and the torch seeds only if torch is
    importable — that way this module is test-importable without torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        # CUDA non-determinism is its own rabbit hole; we don't claim
        # bit-exact reproducibility across GPU vs CPU.
    except ImportError:
        # Torch is only needed at encoding time, not at import time.
        pass


def _load_model(model_name: str, device: str):
    """Lazy-load the SentenceTransformer. Imported here so tests that
    never embed can run with no torch / model download."""
    # Imported locally to keep this module's import surface cheap and
    # to make "no model installed" failures localised.
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name, device=device)
    model.eval()
    return model


def embed_jobs(
    rows: Sequence[JobRow],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
    show_progress: bool = False,
) -> EmbeddingResult:
    """Encode a sequence of :class:`JobRow` into 384-d float32 vectors.

    Empty input returns an empty EmbeddingResult.

    Parameters
    ----------
    rows:
        Jobs to embed. Each row's id becomes its join key in the output.
        Rows whose embedding text is empty (e.g. only missing fields)
        are still embedded — they get a near-zero vector from the
        model — but their ``id`` is still recorded so the output is
        1-to-1 with the input. Callers that want to drop empty-text
        rows should do so via :func:`jobs_loader.validate_rows` first.
    model_name:
        Hugging Face model id. Default per spec.
    batch_size:
        Forward-pass batch size. 64 is comfortable on CPU; raise to
        128–256 if you have GPU memory.
    seed:
        Python + NumPy + (if loaded) torch seed.
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"`` — anything SentenceTransformers
        accepts. Default ``"cpu"`` to avoid requiring GPU at test time.
    show_progress:
        If True, show a tqdm bar during encoding.

    Returns
    -------
    EmbeddingResult with parallel ``ids`` and ``vectors``.
    """
    if not rows:
        return EmbeddingResult(
            ids=[],
            vectors=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            model_name=model_name,
            schema_version=EMBEDDING_SCHEMA_VERSION,
        )

    _seed_everything(seed)

    texts: list[str] = [compose_embedding_text(r) for r in rows]
    ids: list[str] = [str(r.id) for r in rows]

    model = _load_model(model_name=model_name, device=device)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # unit vectors -> dot product == cosine
    )

    # Defensive cast even though encode(..., convert_to_numpy=True)
    # already returns float32 — costs nothing if it's already correct.
    vectors = vectors.astype(np.float32, copy=False)

    if vectors.shape != (len(rows), EMBEDDING_DIM):
        raise RuntimeError(
            f"Encoder returned wrong shape: expected "
            f"({len(rows)}, {EMBEDDING_DIM}), got {vectors.shape!r}"
        )

    return EmbeddingResult(
        ids=ids,
        vectors=vectors,
        model_name=model_name,
        schema_version=EMBEDDING_SCHEMA_VERSION,
    )


__all__ = [
    "EMBEDDING_SCHEMA_VERSION",
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIM",
    "EmbeddingResult",
    "compose_embedding_text",
    "embed_jobs",
]
