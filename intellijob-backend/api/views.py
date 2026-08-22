"""
IntelliJob — DRF views.

Exposes one endpoint:

    POST /api/analyze/
        multipart/form-data with:
            file          -- the candidate's PDF resume        (required)
            target_title  -- the role they want to target      (optional)
        returns JSON:
            extracted_skills  -- sorted list[str] from the resume
            career_mode       -- "targeted" | "discovery"
            career_pathways   -- list[dict] top recommended career pathways
                                 (each with its own matches + roadmap)
            matches           -- top-k matched job specs
            roadmap           -- SkillGapRoadmap dict

Two modes:

* **Targeted** (``target_title`` given): behaves like the original tool —
  embed title+skills, rank jobs against it, single roadmap.
* **Discovery** (``target_title`` blank): treat the resume as the query,
  rank the whole career-pathway taxonomy against it, and return the top
  recommended pathways — each self-contained with its own matches and
  strategic roadmap so the frontend can switch between them.

The view deliberately wires through the existing service modules — no
business logic lives here.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from api.services import matcher, pathways
from api.services.learning_agent import search_learning_resources
from api.services.roadmap_generator import generate_phase_plan, generate_roadmap
from api.services.skill_extractor import (
    extract_skills_from_text,
    extract_text_from_pdf,
)

log = logging.getLogger(__name__)

#: Top-k job specs per career pathway (3-5, chosen by the spec).
DEFAULT_TOP_K = 5

#: Discovery-mode pathway filtering.
MIN_PATHWAY_SCORE = 0.30
MIN_PATHWAY_COVERAGE = 0.15
MAX_PATHWAYS = 8
MIN_PATHWAYS_FLOOR = 3


class AnalyzeView(APIView):
    """Extract skills from a PDF resume and map it onto job matches and
    career pathways."""

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
            resume_file = request.FILES.get("file")
            target_title = (request.data.get("target_title") or "").strip()

            if resume_file is None:
                return Response(
                    {"error": "Missing 'file' (multipart upload)"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # 1. PDF -> Markdown text -> skills
            try:
                pdf_bytes = resume_file.read()
                resume_text = extract_text_from_pdf(pdf_bytes)
                skills = extract_skills_from_text(resume_text)
            except Exception as e:
                log.exception("PDF parse / skill extraction failed")
                return Response(
                    {"error": f"Could not parse PDF: {e!s}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # 2. Embed the query with candidate background context
            try:
                index = matcher.load_job_index()
                query_vec = matcher.embed_query(
                    skills, target_title, resume_context=resume_text
                )
            except FileNotFoundError as e:
                log.error("Job index missing: %s", e)
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            except Exception as e:
                log.exception("Matching failed")
                return Response(
                    {"error": f"Matching failed: {e!s}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            if target_title:
                return self._targeted_response(skills, query_vec, index)

            return self._discovery_response(skills, query_vec, index)

    # ------------------------------------------------------------------- #
    # Mode helpers
    # ------------------------------------------------------------------- #

    def _targeted_response(self, skills, query_vec, index):
        """Classic single-role mode: rank jobs against title+skills, one
        roadmap."""
        try:
            matches = matcher.match_jobs(
                query_vec, top_k=DEFAULT_TOP_K, index=index,
            )
        except Exception as e:
            log.exception("Matching failed")
            return Response(
                {"error": f"Matching failed: {e!s}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # generate_roadmap never raises: falls back deterministically.
        roadmap = generate_roadmap(skills, matches)

        return Response(
            {
                "extracted_skills": skills,
                "career_mode": "targeted",
                "career_pathways": [],
                "matches": [m.to_dict() for m in matches],
                "roadmap": roadmap.to_dict(),
            },
            status=status.HTTP_200_OK,
        )

    def _discovery_response(self, skills, query_vec, index):
        """Career-discovery mode: rank the pathway taxonomy, then produce
        matches + a strategic roadmap for each recommended pathway."""
        pathway_index = pathways.build_pathway_index(index)
        # Get all pathways ranked (no top_k limit) so we can filter by threshold
        ranked = pathways.match_pathways(
            query_vec,
            top_k=len(pathway_index.skill_sets),  # all pathways
            index=index,
            pathway_index=pathway_index,
            skills=skills,
            exclude_fallback=True,
        )

        # Filter by score + coverage thresholds
        filtered = [
            pw for pw in ranked
            if pw.score >= MIN_PATHWAY_SCORE and pw.coverage >= MIN_PATHWAY_COVERAGE
        ]

        # Ensure at least MIN_PATHWAYS_FLOOR visible, cap at MAX_PATHWAYS
        if len(filtered) < MIN_PATHWAYS_FLOOR:
            visible = ranked[:MIN_PATHWAYS_FLOOR]
        else:
            visible = filtered[:MAX_PATHWAYS]

        # Additional pathways for "Show more" section
        additional = [pw for pw in ranked if pw not in visible]

        career_pathways: list[dict] = []
        career_pathways_additional: list[dict] = []

        def _build(pw):
            pw_matches = pathways.match_jobs_in_pathway(
                query_vec, pw.key, top_k=DEFAULT_TOP_K,
                index=index, pathway_index=pathway_index,
            )
            roadmap = generate_roadmap(skills, pw_matches)
            return {
                **pw.to_dict(),
                "matches": [m.to_dict() for m in pw_matches],
                "roadmap": roadmap.to_dict(),
            }

        # Build visible pathways (parallel)
        max_workers = min(MAX_PATHWAYS, len(visible)) if visible else 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            career_pathways = list(pool.map(_build, visible))

        # Build additional pathways (sequential, lighter - no roadmap needed for "Show more")
        for pw in additional:
            career_pathways_additional.append({
                **pw.to_dict(),
                "matches": [],  # not needed for collapsed view
                "roadmap": None,
            })

        # Top-level matches/roadmap mirror the #1 pathway
        if career_pathways:
            top_matches = career_pathways[0]["matches"]
            roadmap = career_pathways[0]["roadmap"]
        else:
            fallback_matches = matcher.match_jobs(
                query_vec, top_k=DEFAULT_TOP_K, index=index,
            )
            top_matches = [m.to_dict() for m in fallback_matches]
            roadmap = generate_roadmap(skills, fallback_matches).to_dict()

        return Response(
            {
                "extracted_skills": skills,
                "career_mode": "discovery",
                "career_pathways": career_pathways,
                "career_pathways_additional": career_pathways_additional,
                "matches": top_matches,
                "roadmap": roadmap,
            },
            status=status.HTTP_200_OK,
        )


class LearningResourcesView(APIView):
    """Fetch curated learning resources for a skill using Tavily search."""

    def get(self, request, *args, **kwargs):
        skill = (request.query_params.get("skill") or "").strip()
        if not skill:
            return Response(
                {"error": "Missing 'skill' query parameter"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            resources = search_learning_resources(skill)
            return Response(
                {"skill": skill, "resources": resources},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            log.exception("Learning resource search failed")
            return Response(
                {"error": f"Resource search failed: {e!s}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class PhasePlanView(APIView):
    """Generate a personalized phase action plan using the LLM + Tavily."""

    def post(self, request, *args, **kwargs):
        skills = request.data.get("skills", [])
        target_title = (request.data.get("target_title") or "").strip()
        phase_name = (request.data.get("phase_name") or "").strip()
        phase_focus = (request.data.get("phase_focus") or "").strip()
        matched_jobs = request.data.get("matched_jobs", [])

        if not phase_name or not phase_focus:
            return Response(
                {"error": "Missing 'phase_name' or 'phase_focus'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Load job index to convert matched_jobs titles to MatchResult objects
            # For now, we'll pass empty matches - the prompt only uses them for context
            index = matcher.load_job_index()
            matches = []

            plan = generate_phase_plan(
                skills=skills,
                target_title=target_title,
                phase_name=phase_name,
                phase_focus=phase_focus,
                matches=matches,
            )
            return Response(
                plan.to_dict(),
                status=status.HTTP_200_OK,
            )
        except FileNotFoundError as e:
            log.error("Job index missing: %s", e)
            return Response(
                {"error": str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            log.exception("Phase plan generation failed")
            return Response(
                {"error": f"Phase plan generation failed: {e!s}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )