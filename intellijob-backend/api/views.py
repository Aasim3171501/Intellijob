"""
IntelliJob — DRF views.

Phase B-2 exposes one endpoint:

    POST /api/analyze/
        multipart/form-data with:
            file          -- the candidate's PDF resume
            target_title  -- the role they want to target
        returns JSON:
            extracted_skills       -- sorted list[str] from the resume
            matches                -- list[dict] top-k matched job specs
            roadmap                -- placeholder dict; Phase B-3 will
                                      swap in a real LangChain + Ollama
                                      roadmap

The view deliberately wires through the existing service modules —
no business logic lives here. This keeps it small, testable, and
trivial to replace with a class-based DRF view if needed later.
"""

from __future__ import annotations

import logging

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from api.services.matcher import match_from_text
from api.services.skill_extractor import extract_skills_from_pdf

log = logging.getLogger(__name__)

# Top-k is spec'd at 3-5. We pick 5 to give the dashboard enough
# context. The frontend can downsample.
DEFAULT_TOP_K = 5


class AnalyzeView(APIView):
    """Extract skills from a PDF resume and match against the job dataset."""

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        resume_file = request.FILES.get("file")
        target_title = (request.data.get("target_title") or "").strip()

        if resume_file is None:
            return Response(
                {"error": "Missing 'file' (multipart upload)"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not target_title:
            return Response(
                {"error": "Missing 'target_title' (non-empty string)"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 1. PDF -> skills.
        try:
            pdf_bytes = resume_file.read()
            skills = extract_skills_from_pdf(pdf_bytes)
        except Exception as e:  # noqa: BLE001
            log.exception("PDF parse / skill extraction failed")
            return Response(
                {"error": f"Could not parse PDF: {e!s}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2. Skills + target_title -> top-k matched jobs.
        try:
            matches = match_from_text(
                skills=skills, target_title=target_title, top_k=DEFAULT_TOP_K,
            )
        except FileNotFoundError as e:
            # Dataset missing — explicit, helpful error.
            log.error("Job index missing: %s", e)
            return Response(
                {"error": str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Matching failed")
            return Response(
                {"error": f"Matching failed: {e!s}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 3. Placeholder roadmap. Phase B-3 will replace this with a
        #    real LangChain + Ollama roadmap. The shape is stable so
        #    the frontend can already render against it.
        roadmap = {
            "status": "pending",
            "note": "Roadmap generation is planned for Phase B-3 "
                    "(LangChain + Ollama).",
            "extracted_skill_count": len(skills),
        }

        return Response(
            {
                "extracted_skills": skills,
                "matches": [m.to_dict() for m in matches],
                "roadmap": roadmap,
            },
            status=status.HTTP_200_OK,
        )