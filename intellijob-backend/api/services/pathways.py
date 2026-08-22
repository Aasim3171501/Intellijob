"""
IntelliJob — Career Pathway & Pathfinding Engine.

This module turns the flat job dataset into a small taxonomy of career
pathways ("Embedded Systems", "AI & Machine Learning", "Software
Engineering", ...) so that a resume can be mapped onto *career
trajectories*, not just individual job specs.

Pipeline:

    JobIndex (ids + vectors + rows)
        ├─ classify(title) -> pathway key       (keyword taxonomy)
        ├─ build_pathway_index() -> centroids   (mean unit vector per pathway)
        └─ match_pathways(query_vec) -> ranked  (cosine of resume vs centroid)
             └─ match_jobs_in_pathway() -> top-k jobs inside that pathway

Design:

* **Keyword taxonomy, not embeddings, for classification.** Job titles
  are short and noisy; a curated keyword list (checked case-insensitively,
  first-match-wins so the taxonomy is a stable priority order) gives
  deterministic, explainable grouping. The *ranking* between pathways
  uses real embeddings: each pathway's centroid is the normalised mean
  of its member job vectors, so similarity is meaningful in the trained
  space.
* **First match wins.** Pathways are ordered most-specific first (e.g.
  "Embedded Systems" before the general "Software Engineering" fallback).
  A title like "Embedded Software Engineer" is embedded/systems, not
  generic software.
* **Self-contained.** Nothing here loads models itself; callers pass a
  JobIndex so tests can feed tiny fixtures and the view can reuse the
  index it already loaded.
* **Deterministic.** The same title always maps to the same pathway; the
  taxonomy is a module-level constant, so behaviour is stable across
  calls and processes.

Public API:
    Pathway                - taxonomy entry (key, name, description, keywords)
    CAREER_PATHWAYS        - ordered taxonomy (most specific first)
    classify(title)        - str, pathway key for a job title
    PathwayIndex           - centroids + member positions per key
    build_pathway_index()  - JobIndex -> PathwayIndex
    match_pathways()       - query_vec -> list[PathwayMatch] (top-k)
    match_jobs_in_pathway() - query_vec + key -> list[MatchResult] (top-k)
    PathwayMatch           - one ranked career pathway (JSON-able)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from api.services.matcher import (
    JobIndex,
    MatchResult,
    load_job_index,
    matches_for_positions,
)
from api.services.skill_extractor import SKILL_PATTERNS, pattern_surface

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pathway:
    """One career pathway in the taxonomy.

    ``keywords`` are matched case-insensitively as substrings of the
    lower-cased job title. First matching pathway (in :data:`CAREER_PATHWAYS`
    order) wins.
    """

    key: str
    name: str
    description: str
    keywords: tuple[str, ...]


#: Ordered taxonomy, most-specific first. The final entry
#: (``software-engineering``) has no keywords and acts as the catch-all.
CAREER_PATHWAYS: tuple[Pathway, ...] = (
    Pathway(
        key="embedded",
        name="Embedded Systems",
        description="Firmware, RTOS, FPGA and hardware-adjacent software roles.",
        keywords=(
            "embedded", "firmware", "fpga", "vhdl", "verilog", "rtos",
            "microcontroller", "iot", "hardware", "robotics", "control systems",
        ),
    ),
    Pathway(
        key="security",
        name="Cyber Security",
        description="Application security, SOC, threat and compliance engineering.",
        keywords=(
            "cyber", "security", "infosec", "soc", "penetration", "threat",
            "vulnerab", "compliance", "cryptograph",
        ),
    ),
    Pathway(
        key="ml-ai",
        name="AI & Machine Learning",
        description="LLMs, computer vision, MLOps and applied research roles.",
        keywords=(
            "data science", "data scientist", "machine learning", "deep learning",
            "llm", "nlp", "recommender", "computer vision", "genai", "generative",
            "applied ai", "ai engineer", "ai research", "mlops", "ml engineer",
        ),
    ),
    Pathway(
        key="data-eng",
        name="Data Engineering",
        description="ETL, warehouses, streaming and data platform roles.",
        keywords=(
            "data engineer", "etl", "data platform", "big data", "data infrastructure",
            "spark", "hadoop", "airflow", "databricks", "snowflake",
            "data warehouse", "kafka", "data pipeline", "data lake",
        ),
    ),
    Pathway(
        key="data-analytics",
        name="Data Analytics",
        description="BI, reporting, quantitative and business analytics roles.",
        keywords=(
            "data analyst", "analytics", "business intelligence", "tableau",
            "power bi", "reporting", "insight", "quantitative", " quant,",
            "business analyst", "revenue analyst",
        ),
    ),
    Pathway(
        key="frontend",
        name="Frontend Engineering",
        description="React, Vue, Angular and UI/UX engineering roles.",
        keywords=(
            "frontend", "front-end", "front end", "react", "vue", "angular",
            "ui engineer", "ux engineer", "web developer", "javascript",
            "css", "design engineer",
        ),
    ),
    Pathway(
        key="backend",
        name="Backend Engineering",
        description="APIs, services, databases and distributed systems roles.",
        keywords=(
            "backend", "back-end", "back end", "api engineer", "server engineer",
            "node", "java", "spring", ".net", "php", "scala", "kotlin",
            "python", "golang", "c++", "c#", "microservices",
            "distributed systems",
        ),
    ),
    Pathway(
        key="fullstack",
        name="Full Stack Engineering",
        description="Full surface web engineering across client and server.",
        keywords=("full stack", "full-stack", "fullstack"),
    ),
    Pathway(
        key="mobile",
        name="Mobile Engineering",
        description="iOS, Android and cross-platform app roles.",
        keywords=("mobile", "ios", "android", "react native", "flutter"),
    ),
    Pathway(
        key="game",
        name="Game Development",
        description="Gameplay, graphics and rendering engine roles.",
        keywords=("game", "graphics engineer", "graphics programmer", "gameplay programmer",
          "rendering", "unreal", "unity", "shader"),
    ),
    Pathway(
        key="devops",
        name="DevOps & Cloud",
        description="SRE, platform, infrastructure and CI/CD engineering.",
        keywords=(
            "devops", "sre", "site reliability", "platform engineer",
            "infrastructure", "cloud engineer", "cloud platform", "ci/cd",
            "sysadmin", "kubernetes", "k8s", "terraform", "linux",
            "release engineer",
        ),
    ),
    Pathway(
        key="qa",
        name="QA & Test Automation",
        description="Software testing, quality assurance and SDET roles.",
        keywords=(
            "qa", "test engineer", "test automation", "quality assurance",
            "sdet", "quality engineer", "manual tester",
        ),
    ),
    Pathway(
        key="architecture",
        name="Solution Architecture",
        description="Enterprise, solution and systems architecture roles.",
        keywords=("architect",),
    ),
    Pathway(
        key="delivery",
        name="Delivery & Product",
        description="Product, programme and delivery management roles.",
        keywords=(
            "project manager", "programme manager", "program manager",
            "product manager", "delivery", "scrum", "agile", "consultant",
            "technical account manager", "engagement", "solution owner",
        ),
    ),
    Pathway(
        key="software-engineering",
        name="Software Engineering",
        description="General software engineering and development roles.",
        keywords=(),
    ),
)

#: Catch-all pathway key (the final, keyword-less entry).
FALLBACK_PATHWAY_KEY: str = "software-engineering"

#: Weight of explicit skill-overlap in the blended pathway score. A pure
#: embedding query dilutes on long resumes (paragraph prose dominates the
#: sentence vector), so the resume's *actual* extracted skills must be the
#: dominant, explainable signal. Must stay > SIM_WEIGHT to guarantee the
#: ranking is driven by the candidate's skills, not embedding noise.
SKILL_WEIGHT: float = 0.6

#: Weight of cosine similarity to the pathway centroid in the blended score.
SIM_WEIGHT: float = 0.4

# --------------------------------------------------------------------------- #
# Per-pathway skill sets (for the hybrid ranking)
# --------------------------------------------------------------------------- #

#: All gazetteer surfaces, longest first, so multi-word phrases like
#: ``machine learning`` match before their single-word parts.
_SURFACES: tuple[str, ...] = tuple(
    sorted({pattern_surface(p) for p in SKILL_PATTERNS}, key=len, reverse=True)
)

#: One combined, word-boundary regex over every gazetteer surface. A single
#: pass over a lower-cased job text collects all skills it mentions, which
#: lets us compute per-pathway skill coverage in O(text) rather than
#: O(patterns * text).
_SKILL_REGEX = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(s) for s in _SURFACES)
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def _scan_skill_tokens(text: str) -> set[str]:
    """Lower-cased gazetteer surfaces mentioned in ``text``."""
    if not text:
        return set()
    return {m.group(0).lower() for m in _SKILL_REGEX.finditer(text)}


def classify(title: str) -> str:
    """Map a job title to a pathway key. First keyword match wins;
    unknown titles fall back to :data:`FALLBACK_PATHWAY_KEY`."""
    t = (title or "").lower()
    for pathway in CAREER_PATHWAYS:
        if any(keyword in t for keyword in pathway.keywords):
            return pathway.key
    return FALLBACK_PATHWAY_KEY


def pathway_by_key(key: str) -> Pathway | None:
    """Look up a Pathway by key, or ``None`` when unknown."""
    for pathway in CAREER_PATHWAYS:
        if pathway.key == key:
            return pathway
    return None


# --------------------------------------------------------------------------- #
# Pathway index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PathwayIndex:
    """Precomputed view of the taxonomy over a JobIndex.

    ``centroids[key]`` is the normalised mean embedding of the pathway's
    member jobs; ``member_positions[key]`` is a tuple of row indices in
    the JobIndex that belong to that pathway; ``skill_sets[key]`` is the
    lower-cased gazetteer skill surface set found across that pathway's
    member job text (title + description), used by the hybrid ranking to
    measure how many of a candidate's extracted skills each pathway
    actually requires.
    """

    centroids: dict[str, np.ndarray]
    member_positions: dict[str, tuple[int, ...]]
    dimension: int
    skill_sets: dict[str, frozenset[str]] = field(default_factory=dict)


def build_pathway_index(index: JobIndex) -> PathwayIndex:
    """Assign every job in ``index`` to a pathway and compute each
    pathway's centroid (normalised mean of its member vectors)."""
    dimension = int(index.vectors.shape[1])

    assignments: list[str] = []
    for job_id in index.ids:
        row = index.rows.get(str(job_id))
        assignments.append(classify(row.title if row else ""))

    assign_arr = np.asarray(assignments, dtype=object)
    centroids: dict[str, np.ndarray] = {}
    member_positions: dict[str, tuple[int, ...]] = {}
    skill_sets: dict[str, frozenset[str]] = {}

    for pathway in CAREER_PATHWAYS:
        key = pathway.key
        positions = np.flatnonzero(assign_arr == key)
        member_positions[key] = tuple(int(i) for i in positions)
        if positions.size == 0:
            centroids[key] = np.zeros(dimension, dtype=np.float32)
            skill_sets[key] = frozenset()
            continue
        centroid = index.vectors[positions].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        centroids[key] = centroid.astype(np.float32, copy=False)

        tokens: set[str] = set()
        for pos in positions:
            row = index.rows.get(str(index.ids[int(pos)]))
            if row is None:
                continue
            text = f"{row.title or ''} {row.description or ''}"
            if text:
                tokens |= _scan_skill_tokens(text)
        skill_sets[key] = frozenset(tokens)

    return PathwayIndex(
        centroids=centroids,
        member_positions=member_positions,
        dimension=dimension,
        skill_sets=skill_sets,
    )


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PathwayMatch:
    """One career pathway ranked against the candidate query.

    ``similarity`` is the raw cosine to the pathway centroid. When the
    ranking is skill-aware (candidate skills passed to
    :func:`match_pathways`), ``matched_skills`` is the subset of the
    candidate's skills that the pathway's jobs actually require,
    ``coverage`` is ``|matched_skills| / |skills|``, and ``score`` is the
    blended ``SIM_WEIGHT * cosine + SKILL_WEIGHT * coverage`` value used
    for display/ordering.
    """

    key: str
    name: str
    description: str
    similarity: float
    match_count: int
    matched_skills: list[str] = field(default_factory=list)
    coverage: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "similarity_score": round(float(self.similarity), 6),
            "score": round(float(self.score), 6),
            "coverage": round(float(self.coverage), 6),
            "matched_skills": list(self.matched_skills),
            "match_count": self.match_count,
        }


def match_pathways(
    query_vec: np.ndarray,
    *,
    top_k: int = 3,
    index: JobIndex | None = None,
    pathway_index: PathwayIndex | None = None,
    skills: Sequence[str] | None = None,
    exclude_fallback: bool = False,
) -> list[PathwayMatch]:
    """Rank every career pathway against the candidate.

    When ``skills`` is provided the ranking is *hybrid*: the cosine
    similarity to the pathway centroid is blended with the fraction of
    the candidate's extracted skills that the pathway's own jobs require.
    The skill-overlap term is what makes the recommendation specific to
    the person's resume — two technical resumes with different stacks get
    different pathway rankings even when their paragraph embeddings are
    nearly identical. Without ``skills`` the ranking is pure cosine
    (backward compatible).

    ``exclude_fallback=True`` drops the keyword-less catch-all
    (``software-engineering``) from the ranking. Discovery mode uses this
    because the catch-all is the bucket of *leftover* jobs — its skill
    set is a near-superset of every other pathway's, so it would
    otherwise dominate every resume and drown out the specific career
    directions we actually want to recommend.

    Returns the top-k, sorted by the blended score (or cosine when
    ``skills`` is ``None``).
    """
    if index is None:
        index = load_job_index()
    if pathway_index is None:
        pathway_index = build_pathway_index(index)

    q = query_vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm

    skill_set: set[str] = {s.lower() for s in (skills or []) if s and s.strip()}
    skill_aware = bool(skill_set)

    rows: list[tuple[float, float, list[str], Pathway]] = []
    for pathway in CAREER_PATHWAYS:
        if exclude_fallback and pathway.key == FALLBACK_PATHWAY_KEY:
            continue
        centroid = pathway_index.centroids[pathway.key]
        cos = float(np.dot(centroid, q))
        if skill_aware:
            p_skills = pathway_index.skill_sets.get(pathway.key, frozenset())
            matched = [s for s in skills if s.lower() in p_skills]
            coverage = len(matched) / len(skill_set)
        else:
            matched = []
            coverage = 0.0
        rows.append((cos, coverage, matched, pathway))

    if skill_aware:
        # Normalise the cosine component across pathways to [0, 1] so the
        # blend is scale-comparable with coverage, then order by score.
        lo = min(r[0] for r in rows)
        hi = max(r[0] for r in rows)
        span = (hi - lo) or 1.0
        rows.sort(
            key=lambda r: SIM_WEIGHT * (r[0] - lo) / span + SKILL_WEIGHT * r[1],
            reverse=True,
        )
    else:
        lo = hi = span = 0.0
        rows.sort(key=lambda r: r[0], reverse=True)

    out: list[PathwayMatch] = []
    for cos, coverage, matched, pathway in rows[: max(top_k, 0)]:
        positions = pathway_index.member_positions[pathway.key]
        blended = (
            SIM_WEIGHT * (cos - lo) / span + SKILL_WEIGHT * coverage
            if skill_aware
            else cos
        )
        out.append(
            PathwayMatch(
                key=pathway.key,
                name=pathway.name,
                description=pathway.description,
                similarity=cos,
                match_count=len(positions),
                matched_skills=matched,
                coverage=coverage,
                score=blended,
            )
        )
    return out


def match_jobs_in_pathway(
    query_vec: np.ndarray,
    key: str,
    *,
    top_k: int = 5,
    index: JobIndex | None = None,
    pathway_index: PathwayIndex | None = None,
) -> list[MatchResult]:
    """Rank just the jobs belonging to pathway ``key`` and return the
    top-k as MatchResults."""
    if index is None:
        index = load_job_index()
    if pathway_index is None:
        pathway_index = build_pathway_index(index)

    positions = pathway_index.member_positions.get(key)
    if not positions:
        return []

    q = query_vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm

    pos_arr = np.asarray(positions, dtype=np.intp)
    scores = index.vectors[pos_arr] @ q
    return matches_for_positions(index, pos_arr, scores, top_k=top_k)


__all__ = [
    "CAREER_PATHWAYS",
    "FALLBACK_PATHWAY_KEY",
    "SIM_WEIGHT",
    "SKILL_WEIGHT",
    "Pathway",
    "PathwayIndex",
    "PathwayMatch",
    "build_pathway_index",
    "classify",
    "match_jobs_in_pathway",
    "match_pathways",
    "pathway_by_key",
]