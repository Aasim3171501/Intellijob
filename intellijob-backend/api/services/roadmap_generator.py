"""
IntelliJob — Phase B-3: local RAG roadmap generation (LangChain + Ollama).

Given the candidate's extracted skills and the top matched job specs,
this module prompts a *local* Ollama model (default ``llama3.2``) to
synthesise a structured, Markdown-ready skill-gap roadmap:

    generate_roadmap(skills, matches) -> SkillGapRoadmap

Design:

* **100 % local.** No cloud API. The model runs on the user's machine
  (Ollama). This preserves the dissertation's GDPR data-minimisation
  claim.
* **Bounded context.** The prompt only ever carries the candidate's
  extracted ``SKILL`` tokens plus the top-k matched job titles and
  description excerpts — never the full dataset.
* **Structured output.** We ask the model for strict JSON (Ollama's
  ``format="json"`` mode) shaped to :class:`SkillGapRoadmap`, then
  validate with Pydantic. A dedicated ``parse_roadmap_json`` helper
  keeps parsing unit-testable.
* **Graceful degradation.** If Ollama is not running, times out, or the
  model returns unparseable output, we fall back to a *deterministic*
  roadmap derived straight from the gazetteer + matched specs. The
  generator **never raises**, so ``POST /api/analyze/`` can never 500
  because the local LLM is unavailable.
* **Injectable client.** ``generate_roadmap(..., client=...)`` accepts
  any object with an ``.invoke(text) -> str`` method, so tests mock the
  LLM and stay fast / deterministic / offline.

Public API:
    DEFAULT_MODEL           - str, local Ollama model tag
    MAX_OUTPUT_TOKENS       - int, num_predict bound for the generator
    ROADMAP_PROMPT_TEMPLATE - str, LangChain PromptTemplate text
    CareerPhase             - pydantic phase schema for the trajectory
    SkillGapRoadmap         - pydantic output schema (incl. career_trajectory)
    build_prompt()          - str, the bounded prompt for a request
    parse_roadmap_json()    - dict | None, lenient JSON extraction
    generate_roadmap()      - SkillGapRoadmap, primary entry point
    reset_llm_cache()       - clear the cached OllamaLLM (tests)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from api.services.matcher import MatchResult
from api.services.skill_extractor import SKILL_PATTERNS

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Local model tag. Pull it once with:  ollama pull llama3.2
DEFAULT_MODEL: str = "qwen2.5:7b"

#: Upper bound on generated tokens — keeps a single request bounded and
#: fast on CPU. 900 tokens is comfortable for a strategic, multi-phase
#: career trajectory JSON blob.
MAX_OUTPUT_TOKENS: int = 900

#: Sampling temperature. Low = more deterministic, which suits a
#: data-grounded "roadmap" task better than creative writing.
TEMPERATURE: float = 0.2

#: Number of matched specs we feed into the prompt (top-k from the matcher).
CONTEXT_JOBS_LIMIT: int = 5


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #


class CareerPhase(BaseModel):
    """One phase of a long-term strategic career trajectory."""

    name: str
    focus: str
    objectives: list[str] = Field(default_factory=list)


class SkillGapRoadmap(BaseModel):
    """Structured roadmap returned by the RAG generator.

    ``career_trajectory`` is the strategic, multi-phase career plan
    (e.g. Phase 1 - Entry-Level Readiness, Phase 2 - Mid-Level
    Progression, Phase 3 - Senior Trajectory). The legacy flat fields
    (``skill_gaps`` / ``learning_steps`` / ``estimated_timeline_weeks``)
    are kept so existing consumers keep working, and are also produced by
    the LLM. The rest are filled in by :func:`generate_roadmap` so the
    API response carries provenance (matched jobs) and a machine-readable
    status flag.
    """

    status: str = Field(default="generated", description="generated | fallback")
    skill_gaps: list[str] = Field(default_factory=list)
    learning_steps: list[str] = Field(default_factory=list)
    estimated_timeline_weeks: dict[str, str] = Field(default_factory=dict)
    career_trajectory: list[CareerPhase] = Field(default_factory=list)
    matched_jobs: list[str] = Field(default_factory=list)
    extracted_skill_count: int = Field(default=0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

#: LangChain-style template. Bounded: candidate skills + top-k matched
#: jobs only. Instructs strict JSON with exactly the schema's keys.
ROADMAP_PROMPT_TEMPLATE: str = """You are IntelliJob, a strategic, long-term local career advisor.
Your job is to analyse the candidate's existing skills against the requirements of
matched UK job specifications and produce a STRATEGIC career trajectory — a multi-phase
plan that spans entry-level readiness through to a senior trajectory — rather than just a
short list of task-level fixes.

CANDIDATE SKILLS (already extracted from the candidate's CV):
{skills}

TOP MATCHED JOB SPECIFICATIONS (title | company | similarity | description excerpt):
{jobs}

Produce a career roadmap as STRICT JSON with EXACTLY these keys:
- "skill_gaps": list of strings. Missing/weak technical competencies the candidate
  needs to add, each phrased as a concrete technology or capability.
- "learning_steps": list of strings. Step-by-step, actionable learning objectives
  (ordered) that close the most urgent gaps.
- "estimated_timeline_weeks": object mapping milestone phase name to a weeks span,
  e.g. {{"Phase 1 - Foundations": "2-3 weeks"}}.
- "career_trajectory": array of EXACTLY 3 phase objects, following a strategic
  long-term trajectory:
    Phase 1 - "Entry-Level Readiness": build the baseline skills to become employable
               at entry level.
    Phase 2 - "Mid-Level Progression": deepen and apply the skills in real projects,
               with increasing ownership and scope.
    Phase 3 - "Senior Trajectory": lead work, mentoring, architecture and ambiguity.
  Each phase object has the shape:
      {{"name": "<phase name>", "focus": "<one-sentence focus>", "objectives": ["<objective>", ...]}}
  Objective strings should be concrete, measurable and career-relevant.

Return ONLY the JSON object. No markdown fences, no commentary, no extra keys."""


def _job_context_lines(matches: Sequence[MatchResult]) -> str:
    """Render the top-k matched specs into prompt context lines."""
    lines: list[str] = []
    for i, m in enumerate(matches[:CONTEXT_JOBS_LIMIT], start=1):
        company = (m.company_display_name or "").strip()
        title = (m.title or "").strip()
        header = f"{i}. {title}"
        if company:
            header += f" | {company}"
        header += f" | similarity={m.similarity:.3f}"
        excerpt = (m.description_excerpt or "").strip()
        lines.append(header)
        if excerpt:
            lines.append(f"   {excerpt}")
    return "\n".join(lines) if lines else "(no matched jobs available)"


def build_prompt(skills: Sequence[str], matches: Sequence[MatchResult]) -> str:
    """Build the bounded prompt for a single roadmap request."""
    skills_text = ", ".join(s for s in (skills or []) if s and s.strip()) or "(none)"
    jobs_text = _job_context_lines(matches or [])
    return ROADMAP_PROMPT_TEMPLATE.format(skills=skills_text, jobs=jobs_text)


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*")


def _extract_json_blob(raw: str) -> str:
    """Strip code fences and surrounding prose, keep the JSON object."""
    text = raw.strip()
    text = _FENCE_RE.sub("", text)
    text = re.sub(r"```\s*$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def parse_roadmap_json(raw: str) -> dict[str, Any] | None:
    """Leniently extract + parse a roadmap JSON object from LLM output.

    Returns ``None`` when the text contains no parseable JSON object.
    """
    blob = _extract_json_blob(raw)
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


# --------------------------------------------------------------------------- #
# Fallback (deterministic, offline-safe)
# --------------------------------------------------------------------------- #

_TIMELINE_FALLBACK: dict[str, str] = {
    "Phase 1 - Entry-Level Readiness": "months 1-3",
    "Phase 2 - Mid-Level Progression": "months 3-9",
    "Phase 3 - Senior Trajectory": "months 9-24",
}


def _fallback_trajectory(
    gaps: Sequence[str],
    endgame_objectives: Sequence[str] = (
        "Lead end-to-end delivery of a cross-team initiative",
        "Mentor junior engineers and set technical standards",
        "Own architectural and hiring decisions",
    ),
) -> list[CareerPhase]:
    """Deterministic three-phase strategic trajectory derived from the
    detected skill gaps. Used when the LLM is unavailable."""
    objectives = [g for g in (gaps or []) if g]
    return [
        CareerPhase(
            name="Phase 1 - Entry-Level Readiness",
            focus="Build the baseline competencies the matched roles demand.",
            objectives=(
                list(objectives[:3])
                or ["Master the core language/stack of the target roles."]
            ),
        ),
        CareerPhase(
            name="Phase 2 - Mid-Level Progression",
            focus="Apply the new skills in real, portfolio-grade projects.",
            objectives=[
                f"Ship a portfolio project using {o}." for o in (objectives[:3] or ["your core stack"])
            ],
        ),
        CareerPhase(
            name="Phase 3 - Senior Trajectory",
            focus="Move from executing to leading: ownership, mentoring, architecture.",
            objectives=list(endgame_objectives),
        ),
    ]


def _detect_required_skills(matches: Sequence[MatchResult]) -> list[str]:
    """Skills the matched specs demand, found via the SKILL gazetteer.

    Cheap substring scan over title + description excerpt using the same
    surface forms as the spaCy ``EntityRuler`` — no model load needed,
    fully deterministic. Case-insensitive; canonical casing preserved.
    """
    haystack = " ".join(
        (m.title or "") + " " + (m.description_excerpt or "") for m in matches
    ).lower()
    found: list[str] = []
    for pattern in SKILL_PATTERNS:
        surface = pattern["pattern"][0]["TEXT"]
        if surface.lower() in haystack:
            found.append(surface)
    # Keep gazetteer order (dedup already guaranteed by the gazetteer tests).
    return found


def _fallback_roadmap(
    skills: Sequence[str],
    matches: Sequence[MatchResult],
    reason: str,
) -> SkillGapRoadmap:
    """Deterministic roadmap used when the LLM is unavailable/unparseable."""
    candidate = {s.lower() for s in (skills or []) if s and s.strip()}
    required = _detect_required_skills(matches)
    gaps = [s for s in required if s.lower() not in candidate] or [
        f"Deepen: {s}" for s in (skills or [])[:3]
    ]

    steps = [f"Learn {gap} and apply it in a small portfolio project." for gap in gaps]
    steps.append(
        "Re-run this analysis with an updated CV to verify the gaps have closed."
    )
    if not steps:
        steps = [
            "Pick one matched job spec and build its core stack end to end.",
            "Re-run this analysis with an updated CV to verify progress.",
        ]

    log.warning("Roadmap fallback used: %s", reason)
    return SkillGapRoadmap(
        status="fallback",
        skill_gaps=gaps,
        learning_steps=steps,
        estimated_timeline_weeks=dict(_TIMELINE_FALLBACK),
        career_trajectory=_fallback_trajectory(gaps),
        matched_jobs=[m.title for m in matches],
        extracted_skill_count=len(skills or []),
    )


# --------------------------------------------------------------------------- #
# Client (lazy, cached)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _get_llm():
    """Lazy, cached OllamaLLM. Imported lazily so this module can be
    imported in environments without langchain_ollama installed, and
    tests never pay the client-construction cost.

    ``validate_model_on_init=False`` defers the model check to invoke
    time — so construction never hits the network; if Ollama is down
    the failure happens inside :func:`generate_roadmap`'s try block.
    """
    from langchain_ollama import OllamaLLM  # type: ignore

    return OllamaLLM(
        model=DEFAULT_MODEL,
        format="json",
        num_predict=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        validate_model_on_init=False,
    )


def reset_llm_cache() -> None:
    """Clear the cached OllamaLLM — used by tests after patching."""
    _get_llm.cache_clear()


# --------------------------------------------------------------------------- #
# Primary entry point
# --------------------------------------------------------------------------- #


def generate_roadmap(
    skills: Sequence[str],
    matches: Sequence[MatchResult],
    *,
    client: Any | None = None,
) -> SkillGapRoadmap:
    """Generate a skill-gap roadmap for a candidate. Never raises.

    Parameters
    ----------
    skills:
        Extracted SKILL tokens from the candidate's CV.
    matches:
        Top-k :class:`MatchResult` objects from the matcher.
    client:
        Injectable LLM object with ``.invoke(text) -> str``. Defaults to
        the cached :func:`_get_llm` (local Ollama). Tests pass a mock.

    Returns
    -------
    SkillGapRoadmap with ``status="generated"`` when the LLM produced
    valid JSON, else ``status="fallback"`` with a deterministic roadmap.
    """
    if client is None:
        try:
            client = _get_llm()
        except Exception as e:  # noqa: BLE001
            log.warning("Ollama client unavailable: %s", e)
            client = None

    if client is None:
        return _fallback_roadmap(skills, matches, reason="ollama client unavailable")

    try:
        raw = client.invoke(build_prompt(skills, matches))
        if not isinstance(raw, str):
            raw = str(raw)
        data = parse_roadmap_json(raw)
        if data is None:
            return _fallback_roadmap(skills, matches, reason="unparseable LLM output")

        partial = SkillGapRoadmap.model_validate(data)
        return partial.model_copy(
            update={
                "status": "generated",
                "matched_jobs": [m.title for m in matches],
                "extracted_skill_count": len(skills or []),
            }
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Roadmap LLM call failed (%s); using fallback", e)
        return _fallback_roadmap(skills, matches, reason=f"LLM error: {e!s}")


__all__ = [
    "CONTEXT_JOBS_LIMIT",
    "DEFAULT_MODEL",
    "MAX_OUTPUT_TOKENS",
    "ROADMAP_PROMPT_TEMPLATE",
    "TEMPERATURE",
    "CareerPhase",
    "SkillGapRoadmap",
    "build_prompt",
    "generate_roadmap",
    "parse_roadmap_json",
    "reset_llm_cache",
]
