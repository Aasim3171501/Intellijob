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
* The skill gazetteer lives in ``skills_gazetteer.json`` next to this
  module — pure data, no code changes to add skills. It is loaded at
  import time into :data:`SKILL_PATTERNS`.
  * ``case_insensitive`` entries match any casing via spaCy's LOWER
    attribute (``python`` matches "python", "Python", "PYTHON").
  * ``case_sensitive`` entries match exact casing via TEXT — reserved
    for terms that are also ordinary English words ("React", "Go",
    "Swift", "Spring") where a lower-case match would be noise.
  * ``phrases`` entries are multi-token skills ("machine learning",
    "C#", "scikit-learn"). Alphanumeric tokens match case-insensitively;
    punctuation tokens (".", "#", "-") match exactly.
* The spaCy nlp object is built lazily and cached at module scope
  via :func:`get_nlp` so importing this module does not pay the
  ~0.3 s model-load cost. Tests can call ``get_nlp.cache_clear()``
  or patch ``_build_nlp``.

Public API:
    SKILL_PATTERNS          - list[dict] for EntityRuler.add_patterns
    DEFAULT_SKILL_LABEL     - "SKILL"
    GAZETTEER_PATH          - path to skills_gazetteer.json
    get_nlp()               - cached spaCy nlp object
    extract_text_from_pdf() - PDF bytes -> Markdown str
    extract_skills_from_text() - str -> list[str]
    extract_skills_from_pdf()  - PDF bytes -> list[str]
"""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------- #
# Skill gazetteer
# --------------------------------------------------------------------------- #

#: The label we stamp on every recognised skill entity.
DEFAULT_SKILL_LABEL: str = "SKILL"

#: Location of the human-maintained gazetteer. Pure JSON — expanding
#: coverage is a data edit, not a code edit.
GAZETTEER_PATH: Path = Path(__file__).with_name("skills_gazetteer.json")


def _load_gazetteer() -> dict:
    """Read and validate the gazetteer JSON file.

    Raises FileNotFoundError / JSONDecodeError if the file is missing
    or malformed — a loud failure is better than silently extracting
    nothing.
    """
    with open(GAZETTEER_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _build_patterns(gazetteer: dict) -> list[dict]:
    """Turn the JSON gazetteer into EntityRuler patterns.

    * ``case_insensitive`` -> ``{"LOWER": ...}`` single-token patterns
    * ``case_sensitive``   -> ``{"TEXT": ...}`` single-token patterns
    * ``phrases``          -> multi-token patterns; alphanumeric tokens
      use LOWER, punctuation tokens (".", "#", "-") use TEXT

    Duplicate surface forms (across categories or match modes) are
    dropped so no skill is matched twice.
    """
    patterns: list[dict] = []
    seen: set[tuple] = set()

    def _add(token_specs: list[dict]) -> None:
        signature = tuple(tuple(sorted(spec.items())) for spec in token_specs)
        if signature in seen:
            return
        seen.add(signature)
        patterns.append({"label": DEFAULT_SKILL_LABEL, "pattern": token_specs})

    for skills in gazetteer.get("case_insensitive", {}).values():
        for skill in skills:
            _add([{"LOWER": skill}])
    for skills in gazetteer.get("case_sensitive", {}).values():
        for skill in skills:
            _add([{"TEXT": skill}])
    for tokens in gazetteer.get("phrases", []):
        specs = [
            {"TEXT": tok} if not tok.isalnum() else {"LOWER": tok}
            for tok in tokens
        ]
        _add(specs)

    return patterns


#: EntityRuler patterns derived from the JSON gazetteer. Kept as the
#: public, inspectable source of truth for tests and other services.
SKILL_PATTERNS: list[dict] = _build_patterns(_load_gazetteer())


def pattern_surface(pattern: dict) -> str:
    """Human-readable surface form of an EntityRuler pattern.

    Single-token patterns return the token value; phrase patterns join
    word tokens with a space (``machine learning``) and glue punctuation
    tokens directly (``c#``, ``scikit-learn``). Case follows the pattern
    (LOWER entries come out lower-cased).
    """
    values = [next(iter(spec.values())) for spec in pattern["pattern"]]
    if any(not v.isalnum() for v in values):
        return "".join(values)
    return " ".join(values)


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
    "DEFAULT_SKILL_LABEL",
    "GAZETTEER_PATH",
    "SKILL_PATTERNS",
    "extract_skills_from_pdf",
    "extract_skills_from_text",
    "extract_text_from_pdf",
    "get_nlp",
    "pattern_surface",
]