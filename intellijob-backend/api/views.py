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

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from api.services import matcher, pathways
from api.services.roadmap_generator import generate_roadmap
from api.services.skill_extractor import extract_skills_from_pdf

log = logging.getLogger(__name__)

#: Top-k job specs per career pathway (3-5, chosen by the spec).
DEFAULT_TOP_K = 5

#: Number of career pathways surfaced in discovery mode.
DEFAULT_PATHWAYS = 3


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

        # 1. PDF -> skills. target_title is optional: a blank value means
        #    "career discovery mode" (match against ALL pathways).
        try:
            pdf_bytes = resume_file.read()
            skills = extract_skills_from_pdf(pdf_bytes)
        except Exception as e:
            log.exception("PDF parse / skill extraction failed")
            return Response(
                {"error": f"Could not parse PDF: {e!s}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2. Embed the query. In targeted mode the title steers the query;
        #    in discovery mode the resume skills alone are the query.
        try:
            index = matcher.load_job_index()
            query_vec = matcher.embed_query(skills, target_title)
        except FileNotFoundError as e:
            # Dataset missing — explicit, helpful error.
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
        ranked = pathways.match_pathways(
            query_vec, top_k=DEFAULT_PATHWAYS, index=index,
            pathway_index=pathway_index,
        )

        career_pathways: list[dict] = []
        for pw in ranked:
            pw_matches = pathways.match_jobs_in_pathway(
                query_vec, pw.key, top_k=DEFAULT_TOP_K,
                index=index, pathway_index=pathway_index,
            )
            roadmap = generate_roadmap(skills, pw_matches)
            career_pathways.append(
                {
                    **pw.to_dict(),
                    "matches": [m.to_dict() for m in pw_matches],
                    "roadmap": roadmap.to_dict(),
                }
            )

        # Top-level matches/roadmap mirror the #1 pathway so consumers
        # that don't understand career_pathways still see something.
        primary = career_pathways[0] if career_pathways else {}

        return Response(
            {
                "extracted_skills": skills,
                "career_mode": "discovery",
                "career_pathways": career_pathways,
                "matches": primary.get("matches", []),
                "roadmap": primary.get(
                    "roadmap",
                    {
                        "status": "fallback",
                        "skill_gaps": [],
                        "learning_steps": [],
                        "estimated_timeline_weeks": {},
                        "career_trajectory": [],
                        "matched_jobs": [],
                        "extracted_skill_count": len(skills or []),
                    },
                ),
            },
            status=status.HTTP_200_OK,
        )