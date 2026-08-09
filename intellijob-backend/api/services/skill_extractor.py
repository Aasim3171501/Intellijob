"""
IntelliJob — Phase B-2: skill extraction from PDF résumés.

Pipeline:

    PDF bytes  ──pymupdf4llm──>  Markdown text
        │
        ▼
    spaCy nlp("text")  ──custom EntityRuler (label=SKILL)──>  set[str] of skills

The ruler is inserted *before* the built-in NER component so our
SKILL labels take precedence over spaCy's generic ORG / PRODUCT
guesses for the same tokens (e.g. "PostgreSQL" should be SKILL, not
ORG; "Python" should be SKILL, not ORG).

Design choices:

* Pure function API. No Django, no DRF. The view layer calls
  :func:`extract_text_from_pdf` and :func:`extract_skills_from_text`
  separately so each is unit-testable in isolation.
* Skills are returned as a sorted, de-duplicated list of strings
  (case-insensitive dedup, original casing preserved).
* The skill gazetteer lives in :data:`SKILL_PATTERNS` at module
  scope. Adding more skills is one edit, no regex changes.
* The spaCy nlp object is built lazily and cached at module scope
  via :func:`get_nlp` so importing this module does not pay the
  ~0.3 s model-load cost. Tests can call ``get_nlp.cache_clear()``
  or patch ``_build_nlp``.

Public API:
    SKILL_PATTERNS          - list[dict] for EntityRuler.add_patterns
    DEFAULT_SKILL_LABEL     - "SKILL"
    get_nlp()               - cached spaCy nlp object
    extract_text_from_pdf() - PDF bytes -> Markdown str
    extract_skills_from_text() - str -> list[str]
    extract_skills_from_pdf()  - PDF bytes -> list[str]
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Iterable

# --------------------------------------------------------------------------- #
# Skill gazetteer
# --------------------------------------------------------------------------- #

#: Surface forms we want to recognise as SKILL tokens.
#: Order does not matter — EntityRuler matches longest-first within
#: the same span. Lower-case surface forms are matched case-
#: insensitively by spaCy's token matcher; we keep the canonical
#: casing in the pattern so the EntityRuler preserves it.
#: Adding new skills: append a dict here. Patterns are deliberately
#: simple substrings (no regex) to keep the test surface small.
SKILL_PATTERNS: list[dict] = [
    # ----- Programming languages -----
    {"label": "SKILL", "pattern": [{"TEXT": "Python"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Java"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "JavaScript"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "TypeScript"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Go"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Rust"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "C++"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "C#"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "SQL"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Bash"}]},

    # ----- Web frameworks -----
    {"label": "SKILL", "pattern": [{"TEXT": "Django"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Flask"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "FastAPI"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "React"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Vue"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Angular"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Next.js"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Node.js"}]},

    # ----- Data / ML -----
    {"label": "SKILL", "pattern": [{"TEXT": "Pandas"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "NumPy"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "scikit-learn"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "TensorFlow"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "PyTorch"}]},

    # ----- Databases / data infra -----
    {"label": "SKILL", "pattern": [{"TEXT": "PostgreSQL"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "MySQL"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "MongoDB"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Redis"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Elasticsearch"}]},

    # ----- DevOps / cloud -----
    {"label": "SKILL", "pattern": [{"TEXT": "Docker"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Kubernetes"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "AWS"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "GCP"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Azure"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Terraform"}]},

    # ----- Version control / CI -----
    {"label": "SKILL", "pattern": [{"TEXT": "Git"}]},

    # ----- Methodologies / soft -----
    {"label": "SKILL", "pattern": [{"TEXT": "Agile"}]},
    {"label": "SKILL", "pattern": [{"TEXT": "Scrum"}]},
]

#: The label we stamp on every recognised skill entity.
DEFAULT_SKILL_LABEL: str = "SKILL"


# --------------------------------------------------------------------------- #
# spaCy model loading
# --------------------------------------------------------------------------- #


def _build_nlp():
    """Build the spaCy pipeline with our EntityRuler prepended to NER.

    We insert before "ner" if present, else prepend. The ruler's
    patterns are stored as-is, so the same SKILL_PATTERNS module
    constant can be inspected by tests without re-loading the model.
    """
    import spacy

    nlp = spacy.load("en_core_web_sm")

    # Use the "entity_ruler" factory so we get the modern spaCy
    # 3.7+ config API. overwrite_ents=True makes SKILL win over
    # the built-in NER's ORG/PRODUCT labels for the same span.
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner" if "ner" in nlp.pipe_names else None,
        config={"overwrite_ents": True},
    )
    ruler.add_patterns(SKILL_PATTERNS)

    return nlp


@lru_cache(maxsize=1)
def get_nlp():
    """Cached spaCy nlp object. Module-level singleton — first call
    pays the model-load cost (~0.3 s), subsequent calls return the
    same pipeline. Tests can ``get_nlp.cache_clear()`` to force a
    rebuild after patching SKILL_PATTERNS.
    """
    return _build_nlp()


# --------------------------------------------------------------------------- #
# PDF text extraction
# --------------------------------------------------------------------------- #


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Convert a PDF byte string to clean Markdown text via pymupdf4llm.

    pymupdf4llm 1.28 accepts a path string (not BytesIO and not
    raw bytes), so we spill the bytes to a tempfile, hand the path
    to pymupdf4llm, and unlink the file on the way out. The temp
    file lives under the system tempdir and is closed before
    pymupdf4llm opens it (Windows requires that).

    Raises whatever pymupdf4llm raises on a malformed PDF — the
    view layer should catch ValueError-shaped errors and turn
    them into HTTP 400.
    """
    import pymupdf4llm

    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        # Write the bytes, close the fd, then call pymupdf4llm on
        # the path. On Windows the fd MUST be closed before the
        # underlying HANDLE is reused by another process.
        with os.fdopen(fd, "wb") as f:
            f.write(pdf_bytes)
        return pymupdf4llm.to_markdown(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Skill extraction
# --------------------------------------------------------------------------- #


def extract_skills_from_text(text: str) -> list[str]:
    """Run spaCy on ``text`` and return recognised SKILL entities.

    Returns a sorted, deduplicated list of skill strings (original
    casing preserved per first occurrence). Whitespace-only or empty
    text returns an empty list.

    The ruler is set with ``overwrite_ents=True`` so SKILL entities
    replace the built-in NER labels for the same span.
    """
    if not text or not text.strip():
        return []

    nlp = get_nlp()
    doc = nlp(text)

    # Preserve insertion order, dedupe case-insensitively.
    seen: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ != DEFAULT_SKILL_LABEL:
            continue
        # ent.text is the surface form (single or multi-token).
        # Strip punctuation that pymupdf4llm sometimes leaves in.
        surface = ent.text.strip().strip(".,;:()[]{}")
        if not surface:
            continue
        key = surface.lower()
        if key not in seen:
            seen[key] = surface

    return sorted(seen.values(), key=str.lower)


def extract_skills_from_pdf(pdf_bytes: bytes) -> list[str]:
    """End-to-end: PDF bytes -> extracted skills."""
    text = extract_text_from_pdf(pdf_bytes)
    return extract_skills_from_text(text)


__all__ = [
    "SKILL_PATTERNS",
    "DEFAULT_SKILL_LABEL",
    "get_nlp",
    "extract_text_from_pdf",
    "extract_skills_from_text",
    "extract_skills_from_pdf",
]