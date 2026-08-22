"""
IntelliJob — Roadmap generation.

Given the candidate's extracted skills and the top matched job specs,
this module synthesises a structured, Markdown-ready skill-gap roadmap:

    generate_roadmap(skills, matches) -> SkillGapRoadmap

Design:

* **Provider-agnostic.** Synthesis delegates to an injectable ``client``
  object exposing ``.invoke(text) -> str``. The default client is an
  adapter for the Ollama Cloud OpenAI-compatible API
  (``https://ollama.com/v1``), configured via the ``OLLAMA_API_KEY`` /
  ``OLLAMA_MODEL`` / ``OLLAMA_BASE_URL`` environment variables. When no
  API key is configured, a *deterministic* roadmap derived straight from
  the gazetteer + matched specs is returned, so ``POST /api/analyze/``
  never 500s and always returns a usable result.
* **Bounded context.** The prompt only ever carries the candidate's
  extracted ``SKILL`` tokens plus the top-k matched job titles and
  description excerpts — never the full dataset.
* **Lenient parsing.** Model output is parsed with :func:`parse_roadmap_json`
  (markdown fences + surrounding prose tolerated) and sanitised with
  :func:`_sanitize_roadmap_data`, so trivial shape quirks (scalar list
  fields, missing fields) don't discard an otherwise-good generation.
* **Bounded latency.** The model call runs in a worker thread with a
  hard timeout, so a slow or stalled connection can never hang the
  request.

Public API:
    OLLAMA_API_KEY         - str, Ollama Cloud API key (env-overridable)
    OLLAMA_MODEL           - str, cloud model tag (env-overridable)
    OLLAMA_BASE_URL        - str, OpenAI-compatible endpoint base
    MAX_OUTPUT_TOKENS      - int, max_tokens bound for the model call
    TEMPERATURE            - float, sampling temperature
    LLM_TIMEOUT_SECONDS    - float, hard ceiling for a model generation
    ROADMAP_PROMPT_TEMPLATE - str, prompt template text
    CareerPhase            - pydantic phase schema for the trajectory
    LearningStep           - pydantic step schema for the milestones
    SkillGapRoadmap        - pydantic output schema
    build_prompt()         - str, the bounded prompt for a request
    parse_roadmap_json()   - dict | None, lenient JSON extraction
    generate_roadmap()     - SkillGapRoadmap, primary entry point
    reset_llm_cache()      - clear the cached LLM client (tests)
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, field_validator

from api.services.learning_agent import search_learning_resources
from api.services.matcher import MatchResult
from api.services.skill_extractor import SKILL_PATTERNS, pattern_surface

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Ollama Cloud API key (https://ollama.com/settings/keys). When unset,
#: ``generate_roadmap`` falls back to the deterministic roadmap.
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")

#: Cloud model tag, e.g. ``gpt-oss:120b`` or ``qwen3:32b``.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

#: OpenAI-compatible endpoint base URL for the Ollama Cloud API.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")

#: Upper bound on generated tokens. 4096 leaves generous room for a rich,
#: multi-step roadmap (3-phase trajectory + step objects with overviews)
#: so the model never truncates mid-JSON — truncation is the #1 cause of
#: "unparseable LLM output" fallbacks. The call is still hard-bounded by
#: LLM_TIMEOUT_SECONDS, so worst case we fall back deterministically.
MAX_OUTPUT_TOKENS: int = 4096

#: Hard ceiling for a single LLM generation. The model call runs in a
#: worker thread with this timeout, so a slow or stalled network call can
#: never hang an API request indefinitely — worst case we fall back to
#: the deterministic roadmap.
LLM_TIMEOUT_SECONDS: float = 180.0

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
    typical_titles: list[str] = Field(default_factory=list, description="Typical job titles for this phase")
    salary_range_gbp: str = Field(default="", description="UK salary range, e.g. '£45k-£65k'")
    required_skills: list[str] = Field(default_factory=list, description="Core skills expected at this phase")
    next_phase_unlocks: list[str] = Field(default_factory=list, description="What to master to reach next phase")


class PhaseWeek(BaseModel):
    """One week in a phase plan."""

    week: int
    theme: str
    milestones: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    project: str = ""


class PhasePlan(BaseModel):
    """Personalized action plan for a career trajectory phase."""

    phase_name: str
    phase_focus: str
    total_weeks: int
    weeks: list[PhaseWeek] = Field(default_factory=list)
    key_projects: list[str] = Field(default_factory=list)
    common_pitfalls: list[str] = Field(default_factory=list)
    resource_priorities: list[str] = Field(default_factory=list)
    resources: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # skill -> resources

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class LearningStep(BaseModel):
    """One actionable learning milestone with an effort estimate.

    ``id`` is a stable slug (e.g. ``step-1``) so the frontend can key
    components and "explore" modals without relying on text equality.
    ``primary_skill`` is the single technology/concept the step centres
    on — used to look up recommended learning resources.
    """

    id: str
    title: str
    overview: str
    primary_skill: str
    estimated_hours: int = Field(default=20, ge=1)

    @field_validator("estimated_hours", mode="before")
    @classmethod
    def _clamp_min_hours(cls, value: Any) -> Any:
        """Coerce ``0`` / negative hour estimates up to ``1``.

        The local LLM occasionally emits ``0`` for a step's effort.
        Rejecting the whole roadmap for that would silently degrade a good
        generation into the deterministic fallback, so clamp to the
        schema's minimum instead. Non-numeric values are passed through
        so Pydantic still raises its normal type error for them.
        """
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return value


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
    learning_steps: list[LearningStep] = Field(default_factory=list)
    estimated_timeline_weeks: dict[str, str] = Field(default_factory=dict)
    career_trajectory: list[CareerPhase] = Field(default_factory=list)
    matched_jobs: list[str] = Field(default_factory=list)
    extracted_skill_count: int = Field(default=0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

#: Prompt template. Bounded: candidate skills + top-k matched jobs only.
#: Instructs strict JSON with exactly the schema's keys.
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
- "learning_steps": list of step objects, ordered from most urgent to least. Each
  step object has EXACTLY this shape:
      {{"id": "step-1", "title": "Learn Spring Framework through online tutorials",
        "overview": "A concise 2-sentence explanation of why this milestone matters
                     and what core concepts to focus on.",
        "primary_skill": "Spring Framework", "estimated_hours": 20}}
  "id" must be unique per step ("step-1", "step-2", ...). "primary_skill" must be
  a single concrete technology or concept. "estimated_hours" is a positive integer
  (always at least 1, never 0 or negative) estimating the study effort in hours.
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
       {{"name": "<phase name>", "focus": "<one-sentence focus>", "objectives": ["<objective>", ...],
         "typical_titles": ["<title>", ...], "salary_range_gbp": "<UK salary range, e.g. £45k-£65k>",
         "required_skills": ["<skill>", ...], "next_phase_unlocks": ["<skill>", ...]}}
   Objective strings should be concrete, measurable and career-relevant.
   typical_titles: 2-4 common UK job titles for this phase.
   salary_range_gbp: realistic UK range for this phase.
   required_skills: 4-6 core technologies/competencies expected at this phase.
   next_phase_unlocks: 3-5 specific skills/experiences that unlock progression to the next phase.

Return ONLY the JSON object. No markdown fences, no commentary, no extra keys."""


# --------------------------------------------------------------------------- #
# Phase Plan prompt
# --------------------------------------------------------------------------- #

#: Prompt template for generating a personalized phase action plan.
PHASE_PLAN_PROMPT_TEMPLATE: str = """You are IntelliJob, a senior career coach creating a personalized weekly action plan.

CANDIDATE PROFILE:
- Current skills: {skills}
- Target role: {target_title}
- Current career phase: {phase_name}
- Phase focus: {phase_focus}
- Matched job requirements (top matches):
{jobs}

Generate a CONCRETE, week-by-week action plan for this specific career phase. The plan should be tailored to the candidate's existing skills and the requirements of the matched jobs.

Return STRICT JSON with EXACTLY these keys:
- "phase_name": string (the phase name passed in)
- "phase_focus": string (the phase focus passed in)
- "total_weeks": integer (8-12 weeks typical for a phase)
- "weeks": array of week objects, each with:
    {{"week": 1, "theme": "Week theme", "milestones": ["specific milestone 1", "..."], "skills": ["skill1", "skill2"], "project": "portfolio project description"}}
- "key_projects": array of 2-3 strings, each a one-sentence description of a major portfolio project that demonstrates phase readiness
- "common_pitfalls": array of 3-5 specific mistakes candidates make in this phase
- "resource_priorities": array of 3-5 skills/concepts to prioritize learning first

Guidelines:
- Weeks should be sequential and build on each other
- Milestones must be concrete and measurable (e.g. "Deploy a REST API to Kubernetes" not "Learn Kubernetes")
- Projects should be portfolio-ready and relevant to matched job requirements
- Skills per week should be specific technologies/concepts
- Pitfalls should be specific to this phase transition (e.g. "Trying to learn 5 frameworks at once" for Entry-Level)
- Resource priorities should be the highest-impact skills to learn first
- key_projects MUST be an array of strings (one sentence each), NOT objects

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


#: Canonical roadmap keys — used to pick the roadmap object out of a
#: response that may contain nested/other JSON objects.
_ROADMAP_KEYS = frozenset(
    {"skill_gaps", "learning_steps", "estimated_timeline_weeks", "career_trajectory"}
)

#: Wrapper keys the model occasionally emits around the actual roadmap.
_WRAPPER_KEYS = (
    "roadmap",
    "result",
    "data",
    "output",
    "response",
    "plan",
    "career_roadmap",
)


def _roadmap_key_count(data: dict[str, Any]) -> int:
    """Count how many canonical roadmap keys a dict carries."""
    return sum(1 for k in _ROADMAP_KEYS if k in data)


def _collect_json_objects(text: str) -> list[dict[str, Any]]:
    """Parse every complete JSON object found in ``text``.

    ``json.JSONDecoder.raw_decode`` parses the object starting at each
    ``{``, so surrounding prose, markdown fences and nested objects are
    all tolerated without brace-slicing heuristics. Returns the objects in
    document order.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            objects.append(data)
    return objects


def _repair_json_text(text: str) -> str:
    """Fix the most common LLM JSON defect: trailing commas.

    The model occasionally emits ``{"a": [1, 2,], "b": {...},}``. Removing
    any comma that immediately precedes a closing ``}`` / ``]`` makes the
    top-level object parseable again. No-op on well-formed JSON.
    """
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_roadmap_json(raw: str) -> dict[str, Any] | None:
    """Leniently extract + parse a roadmap JSON object from LLM output.

    Handles markdown code fences (`` ```json ... ``` ``), surrounding
    prose, trailing commas, and responses that nest the roadmap inside a
    wrapper object or bury it among other JSON: every complete JSON object
    in the text is parsed and the one carrying the most canonical roadmap
    keys (``skill_gaps`` / ``learning_steps`` / ``estimated_timeline_weeks`` /
    ``career_trajectory``) is returned, so a nested step object can never
    be mistaken for the roadmap.

    Returns ``None`` when the text contains no parseable JSON object.
    """
    if not raw:
        return None
    # Strip any markdown code-fence markers. JSON values never contain
    # triple backticks, so removing them globally is safe.
    text = re.sub(r"```[a-zA-Z]*", "", raw).strip()
    objects = _collect_json_objects(_repair_json_text(text))
    if not objects:
        return None
    return max(objects, key=_roadmap_key_count)


class _RoadmapParseError(ValueError):
    """Raised when model output cannot be turned into a SkillGapRoadmap."""


def _s(value: Any) -> str:
    """Coerce a value to a trimmed string (``None`` -> ``""``)."""
    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    """Coerce a scalar/sequence into a list for the list-typed fields."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        # The model sometimes emits a single string for a list field.
        return [value]
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce to a dict (anything else -> empty dict)."""
    return value if isinstance(value, dict) else {}


def _sanitize_roadmap_data(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM output quirks so a good generation is never
    rejected for a trivial shape mismatch — the #1 cause of silent
    fallbacks. Missing fields get defaults; list fields accept a scalar;
    non-dict steps/phases are dropped rather than failing the whole model.
    """
    gaps = [g for g in (_s(v) for v in _as_list(data.get("skill_gaps"))) if g]

    steps: list[dict[str, Any]] = []
    for i, raw_step in enumerate(_as_list(data.get("learning_steps")), start=1):
        if not isinstance(raw_step, dict):
            continue
        primary = _s(raw_step.get("primary_skill")) or "Core skill"
        hours = raw_step.get("estimated_hours", 20)
        try:
            hours_int = max(int(float(hours)), 1)
        except (TypeError, ValueError):
            hours_int = 20
        steps.append(
            {
                "id": _s(raw_step.get("id")) or f"step-{i}",
                "title": _s(raw_step.get("title")) or f"Learn {primary}",
                "overview": _s(raw_step.get("overview"))
                or f"Build a working foundation in {primary}.",
                "primary_skill": primary,
                "estimated_hours": hours_int,
            }
        )

    phases: list[dict[str, Any]] = []
    for i, raw_phase in enumerate(_as_list(data.get("career_trajectory")), start=1):
        if not isinstance(raw_phase, dict):
            continue
        phases.append(
            {
                "name": _s(raw_phase.get("name")) or f"Phase {i}",
                "focus": _s(raw_phase.get("focus")) or "Continue growing as an engineer.",
                "objectives": [
                    o for o in (_s(v) for v in _as_list(raw_phase.get("objectives"))) if o
                ],
            }
        )

    timeline = _as_dict(data.get("estimated_timeline_weeks"))
    timeline = {k: v for k, v in ((_s(k), _s(v)) for k, v in timeline.items()) if k and v}

    count = data.get("extracted_skill_count", 0)
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 0

    return {
        "status": _s(data.get("status")) or "generated",
        "skill_gaps": gaps,
        "learning_steps": steps,
        "estimated_timeline_weeks": timeline,
        "career_trajectory": phases,
        "matched_jobs": [j for j in (_s(v) for v in _as_list(data.get("matched_jobs"))) if j],
        "extracted_skill_count": max(count, 0),
    }


def _normalize_roadmap_data(data: dict[str, Any]) -> dict[str, Any]:
    """Map the model's common structural deviations onto the roadmap schema.

    The model occasionally ignores the requested key set and emits a
    wrapper object (``{"roadmap": {...}}``) or a different schema
    (observed with gpt-oss:120b: ``role`` / ``overview`` / ``learning_path`` /
    ``milestones`` / ``key_skills`` / ``tools_and_technologies`` /
    ``career_progression``). Without normalisation those would be
    discarded as an empty roadmap, silently degrading a good generation
    into the deterministic fallback. This maps the known alternatives so
    the sanitizer can do its job.
    """
    normalized = dict(data)
    for wrapper in _WRAPPER_KEYS:
        inner = normalized.get(wrapper)
        if isinstance(inner, dict):
            normalized = dict(inner)
            break

    def _pick(*keys: str) -> Any:
        for key in keys:
            if key in normalized:
                return normalized.get(key)
        return None

    if "skill_gaps" not in normalized:
        key_skills = _as_list(_pick("key_skills", "skills"))
        tools = _as_list(_pick("tools_and_technologies", "tools", "technologies"))
        if key_skills or tools:
            normalized["skill_gaps"] = key_skills + [
                t for t in tools if t not in key_skills
            ]

    if "learning_steps" not in normalized:
        learning_path = _pick("learning_path", "steps", "step_plan")
        if learning_path is not None:
            normalized["learning_steps"] = learning_path

    if "estimated_timeline_weeks" not in normalized:
        timeline = _pick("timeline", "milestones", "duration")
        if timeline is not None:
            normalized["estimated_timeline_weeks"] = timeline

    if "career_trajectory" not in normalized:
        progression = _pick(
            "career_progression", "progression", "phases", "career_path"
        )
        if progression is not None:
            normalized["career_trajectory"] = progression

    return normalized


def _coerce_roadmap(raw: str) -> SkillGapRoadmap:
    """Parse + leniently validate raw model text into a SkillGapRoadmap.

    Raises :class:`_RoadmapParseError` when the text contains no JSON
    object, contains no roadmap content (so a nested step object or a
    schema deviation that normalisation couldn't map never silently
    becomes an empty roadmap), or (defensively) fails Pydantic validation.
    """
    data = parse_roadmap_json(raw)
    if data is None:
        raise _RoadmapParseError("no JSON object found in model output")
    data = _normalize_roadmap_data(data)
    sanitized = _sanitize_roadmap_data(data)
    try:
        partial = SkillGapRoadmap.model_validate(sanitized)
    except Exception as e:
        raise _RoadmapParseError(str(e)) from e
    if not (
        partial.skill_gaps
        or partial.learning_steps
        or partial.estimated_timeline_weeks
        or partial.career_trajectory
    ):
        raise _RoadmapParseError(
            f"no roadmap content found in parsed object (keys: {sorted(data)})"
        )
    return partial


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

    Cheap word-boundary scan over title + description excerpt using the
    same surface forms as the spaCy ``EntityRuler`` — no model load
    needed, fully deterministic. Case-insensitive.

    Word boundaries matter: single-letter terms like "R" or "C" would
    otherwise match inside every word ("terraform", "docker", ...).
    """
    import re

    haystack = " ".join(
        (m.title or "") + " " + (m.description_excerpt or "") for m in matches
    ).lower()
    found: list[str] = []
    for pattern in SKILL_PATTERNS:
        surface = pattern_surface(pattern)
        needle = re.escape(surface.lower())
        if re.search(rf"(?<![a-z0-9]){needle}(?![a-z0-9])", haystack):
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

    steps: list[LearningStep] = []
    for i, gap in enumerate(gaps, start=1):
        primary = gap.removeprefix("Deepen: ").strip()
        steps.append(
            LearningStep(
                id=f"step-{i}",
                title=f"Learn {primary} and apply it in a small portfolio project.",
                overview=(
                    f"Build a working foundation in {primary} — understand its core "
                    "concepts, tooling and best practices — then prove mastery by "
                    "shipping a small portfolio piece that uses it."
                ),
                primary_skill=primary,
                estimated_hours=20,
            )
        )
    if not steps:
        steps.append(
            LearningStep(
                id="step-1",
                title="Pick one matched job spec and build its core stack end to end.",
                overview=(
                    "Choose the closest matched role and implement its primary "
                    "technologies in a realistic project to build job-ready experience."
                ),
                primary_skill="Full-stack fundamentals",
                estimated_hours=30,
            )
        )
    steps.append(
        LearningStep(
            id=f"step-{len(steps) + 1}",
            title="Re-run this analysis with an updated CV to verify the gaps have closed.",
            overview=(
                "Re-run IntelliJob after updating your CV to confirm the identified "
                "gaps are now covered and to surface any remaining ones."
            ),
            primary_skill="Career planning",
            estimated_hours=5,
        )
    )

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


def _console_print(text: str) -> None:
    """Print to the console without crashing on non-encodable characters
    (e.g. U+2011 on a cp1252 Windows console). Falls back to an ASCII-safe
    encoding with ``backslashreplace`` so the escape codes stay visible
    (``repr`` is NOT enough: it keeps printable non-ASCII literals)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


class _OllamaRoadmapClient:
    """Thin adapter exposing ``.invoke(text) -> SkillGapRoadmap`` over the
    Ollama Cloud OpenAI-compatible ``/v1/chat/completions`` endpoint.

    ``session`` is injectable so tests can stub the HTTP layer without a
    network round-trip.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = LLM_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        requester = self._session.post if self._session is not None else requests.post
        response = requester(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def invoke(self, text: str) -> SkillGapRoadmap:
        return _coerce_roadmap(self.invoke_raw(text))

    def invoke_raw(self, text: str) -> str:
        """Return raw model output text (no parsing)."""
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": text}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }
        data = self._post(payload)
        content = ""
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            content = ""
        content = content if isinstance(content, str) else str(content)
        _console_print(f"[OLLAMA] {self._model} returned {len(content)} chars:\n{content}")
        log.warning("Ollama Cloud %s returned %d chars", self._model, len(content))
        if not content:
            log.warning(
                "Ollama Cloud response had no text content; full response: %s",
                ascii(data),
            )
        return content


@lru_cache(maxsize=1)
def _get_llm_client() -> Any | None:
    """Lazy, cached LLM client.

    Returns ``None`` (-> deterministic fallback) when ``OLLAMA_API_KEY``
    is missing, so a missing key can never 500 an API request. Reads the
    environment live (not the module-level constants) so ``.env`` edits
    and test patching take effect without a reload.
    """
    api_key = os.getenv("OLLAMA_API_KEY", "")
    if not api_key:
        log.warning("OLLAMA_API_KEY is not set; using deterministic fallback")
        return None
    return _OllamaRoadmapClient(
        api_key=api_key,
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"),
    )


def reset_llm_cache() -> None:
    """Clear the cached LLM client — used by tests after patching."""
    _get_llm_client.cache_clear()


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
        Injectable LLM object with ``.invoke(text) -> str`` (may return
        either raw JSON text or a :class:`SkillGapRoadmap`). Defaults to
        the cached :func:`_get_llm_client`. Tests pass a mock.

    Returns
    -------
    SkillGapRoadmap with ``status="generated"`` when the LLM produced
    valid content, else ``status="fallback"`` with a deterministic roadmap.
    """
    if client is None:
        try:
            client = _get_llm_client()
        except Exception as e:  # noqa: BLE001
            log.warning("LLM client unavailable: %s", e)
            client = None

    if client is None:
        return _fallback_roadmap(
            skills, matches, reason="no LLM client configured (set OLLAMA_API_KEY)"
        )

    # Bounded generation: run the model call in a worker thread with a
    # hard timeout so a slow/stalled network call can NEVER hang the
    # request. Model output is stochastic (schema deviations, malformed
    # JSON, truncation happen occasionally), so a failed attempt is retried
    # once before falling back — a one-off bad generation is the common
    # case, and a single retry recovers it without hurting latency too much.
    last_error = "unknown error"
    for attempt in range(1, 3):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="roadmap-llm")
        try:
            future = pool.submit(client.invoke, build_prompt(skills, matches))
            result = future.result(timeout=LLM_TIMEOUT_SECONDS)
            if isinstance(result, SkillGapRoadmap):
                partial = result
            elif isinstance(result, str):
                # Backward-compatible with duck-typed test/mock clients whose
                # .invoke() returns raw JSON text.
                partial = _coerce_roadmap(result)
            else:
                raise TypeError(
                    f"unexpected client.invoke return type: {type(result).__name__}"
                )

            # An effectively-empty roadmap is as useless as unparseable output —
            # fall back so the UI never renders an empty "No Roadmap Available".
            if not (
                partial.skill_gaps
                or partial.learning_steps
                or partial.estimated_timeline_weeks
                or partial.career_trajectory
            ):
                raise ValueError("model returned an empty roadmap")

            return partial.model_copy(
                update={
                    "status": "generated",
                    "matched_jobs": [m.title for m in matches],
                    "extracted_skill_count": len(skills or []),
                }
            )
        except FuturesTimeoutError:
            last_error = f"LLM timeout after {LLM_TIMEOUT_SECONDS}s"
        except Exception as e:  # noqa: BLE001
            last_error = f"LLM error: {e!s}"
        finally:
            # wait=False: if the generation is still running after a timeout,
            # abandon it in the background rather than blocking the response.
            pool.shutdown(wait=False)
        log.warning(
            "Roadmap LLM attempt %d/2 failed (%s); %s",
            attempt,
            last_error,
            "retrying once" if attempt == 1 else "using fallback",
        )

    return _fallback_roadmap(skills, matches, reason=last_error)


# --------------------------------------------------------------------------- #
# Phase Plan generation
# --------------------------------------------------------------------------- #

def build_phase_plan_prompt(
    skills: Sequence[str],
    target_title: str,
    phase_name: str,
    phase_focus: str,
    matches: Sequence[MatchResult],
) -> str:
    """Build the prompt for a phase plan request."""
    skills_text = ", ".join(s for s in (skills or []) if s and s.strip()) or "(none)"
    jobs_text = _job_context_lines(matches or [])
    return PHASE_PLAN_PROMPT_TEMPLATE.format(
        skills=skills_text,
        target_title=target_title or "(not specified)",
        phase_name=phase_name,
        phase_focus=phase_focus,
        jobs=jobs_text,
    )


# --------------------------------------------------------------------------- #
# Phase Plan JSON parsing (lenient, like parse_roadmap_json)
# --------------------------------------------------------------------------- #

_PHASE_PLAN_KEYS = frozenset(
    {"phase_name", "phase_focus", "total_weeks", "weeks", "key_projects",
     "common_pitfalls", "resource_priorities"}
)

def _normalize_phase_plan_data(data: dict) -> dict:
    """Normalize PhasePlan data - convert key_projects objects to strings."""
    if "key_projects" in data and isinstance(data["key_projects"], list):
        normalized = []
        for item in data["key_projects"]:
            if isinstance(item, dict):
                # Convert {title, description} or {name, description} to string
                if "title" in item:
                    normalized.append(f"{item['title']}: {item.get('description', '')}")
                elif "name" in item:
                    normalized.append(f"{item['name']}: {item.get('description', '')}")
                else:
                    # Fallback: join all values
                    normalized.append(" ".join(str(v) for v in item.values() if v))
            else:
                normalized.append(str(item))
        data["key_projects"] = normalized
    return data


def _coerce_phase_plan(raw: str) -> PhasePlan:
    """Parse leniently and validate into a PhasePlan."""
    import json
    # Try direct parse first
    try:
        data = json.loads(raw)
        data = _normalize_phase_plan_data(data)
        return PhasePlan.model_validate(data)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from markdown fences or surrounding prose
    # Use the same logic as parse_roadmap_json
    text = raw.strip()
    # Remove markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{"):
                try:
                    data = json.loads(part)
                    return PhasePlan.model_validate(data)
                except json.JSONDecodeError:
                    continue

    # Find first complete JSON object
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        if text[idx] == "{":
            try:
                obj, end = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict) and _PHASE_PLAN_KEYS & set(obj.keys()):
                    return PhasePlan.model_validate(obj)
                idx = end
            except json.JSONDecodeError:
                idx += 1
        else:
            idx += 1

    raise ValueError("no valid PhasePlan JSON object found in model output")


def _enrich_plan_with_resources(plan: PhasePlan) -> PhasePlan:
    """Fetch learning resources for skills mentioned in the plan."""
    all_skills = set()
    for week in plan.weeks:
        all_skills.update(week.skills)
    all_skills.update(plan.resource_priorities)

    resources_map: dict[str, list[dict[str, Any]]] = {}
    for skill in all_skills:
        try:
            resources = search_learning_resources(skill, max_results=3)
            if resources:
                resources_map[skill] = resources
        except Exception:
            continue

    return plan.model_copy(update={"resources": resources_map})


def generate_phase_plan(
    skills: Sequence[str],
    target_title: str,
    phase_name: str,
    phase_focus: str,
    matches: Sequence[MatchResult],
    *,
    client: Any | None = None,
) -> PhasePlan:
    """Generate a personalized week-by-week action plan for a career phase.

    Parameters
    ----------
    skills:
        Extracted SKILL tokens from the candidate's CV.
    target_title:
        The role the candidate is targeting.
    phase_name:
        Name of the career trajectory phase (e.g. "Phase 1 - Entry-Level Readiness").
    phase_focus:
        One-sentence focus of the phase.
    matches:
        Top-k :class:`MatchResult` objects from the matcher.
    client:
        Injectable LLM object with ``.invoke(text) -> str``. Defaults to
        the cached :func:`_get_llm_client`.

    Returns
    -------
    PhasePlan with weekly milestones, projects, pitfalls, and resources.
    """
    if client is None:
        try:
            client = _get_llm_client()
        except Exception as e:  # noqa: BLE001
            log.warning("LLM client unavailable: %s", e)
            client = None

    if client is None:
        # Return a minimal deterministic plan
        return PhasePlan(
            phase_name=phase_name,
            phase_focus=phase_focus,
            total_weeks=8,
            weeks=[
                PhaseWeek(
                    week=1,
                    theme="Foundation & Planning",
                    milestones=["Assess current skill gaps", "Create learning schedule"],
                    skills=[],
                    project="",
                )
            ],
            key_projects=["Build a portfolio project demonstrating phase skills"],
            common_pitfalls=["Trying to learn everything at once", "Skipping fundamentals"],
            resource_priorities=list(skills)[:3] if skills else [],
        )

    last_error = "unknown error"
    for attempt in range(1, 3):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phase-plan-llm")
        try:
            prompt = build_phase_plan_prompt(skills, target_title, phase_name, phase_focus, matches)
            future = pool.submit(client.invoke_raw, prompt)
            result = future.result(timeout=LLM_TIMEOUT_SECONDS)

            if isinstance(result, str):
                # Parse JSON from model output using lenient PhasePlan parser
                try:
                    plan = _coerce_phase_plan(result)
                except Exception as e:
                    log.warning("Phase plan JSON parsing failed: %s", e)
                    log.debug("Raw LLM output: %s", result[:2000])
                    raise
            else:
                plan = result

            # Ensure we have a PhasePlan instance
            try:
                if not isinstance(plan, PhasePlan):
                    plan = PhasePlan.model_validate(plan)
            except Exception as e:
                log.warning("PhasePlan validation failed: %s", e)
                raise
            # Enrich with Tavily resources
            try:
                enriched = _enrich_plan_with_resources(plan)
                return enriched
            except Exception as e:
                log.warning("Phase plan enrichment failed: %s", e)
                raise

        except FuturesTimeoutError:
            last_error = f"LLM timeout after {LLM_TIMEOUT_SECONDS}s"
        except Exception as e:  # noqa: BLE001
            last_error = f"LLM error: {e!s}"
        finally:
            pool.shutdown(wait=False)
        log.warning(
            "Phase plan LLM attempt %d/2 failed (%s); %s",
            attempt,
            last_error,
            "retrying once" if attempt == 1 else "using fallback",
        )

    # Fallback deterministic plan
    return PhasePlan(
        phase_name=phase_name,
        phase_focus=phase_focus,
        total_weeks=8,
        weeks=[
            PhaseWeek(
                week=1,
                theme="Foundation & Planning",
                milestones=["Assess current skill gaps", "Create learning schedule"],
                skills=[],
                project="",
            )
        ],
        key_projects=["Build a portfolio project demonstrating phase skills"],
        common_pitfalls=["Trying to learn everything at once", "Skipping fundamentals"],
        resource_priorities=list(skills)[:3] if skills else [],
    )


__all__ = [
    "CONTEXT_JOBS_LIMIT",
    "LLM_TIMEOUT_SECONDS",
    "MAX_OUTPUT_TOKENS",
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "ROADMAP_PROMPT_TEMPLATE",
    "PHASE_PLAN_PROMPT_TEMPLATE",
    "TEMPERATURE",
    "CareerPhase",
    "LearningStep",
    "PhasePlan",
    "PhaseWeek",
    "SkillGapRoadmap",
    "build_prompt",
    "generate_roadmap",
    "generate_phase_plan",
    "parse_roadmap_json",
    "reset_llm_cache",
]
