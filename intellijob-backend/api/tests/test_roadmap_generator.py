"""
Tests for api.services.roadmap_generator.

Strategy:

* The LangChain prompt builder is pure string code — tested directly.
* JSON parsing is a pure function — tested with plain JSON, fenced
  JSON, and prose-embedded JSON.
* generate_roadmap is tested with an injected fake ``client`` object
  (duck-typed ``.invoke(text) -> str``), so no Ollama / network is ever
  touched and the tests stay fast and deterministic.
* Fallback paths are exercised for: unparseable output, invalid schema
  output, client exceptions, and an unavailable client factory.
* Skill-gap fallback detection uses a real MatchResult fixture whose
  description excerpt names technologies from the gazetteer.
"""

from __future__ import annotations

import unittest
from unittest import mock

from api.services.matcher import MatchResult
from api.services.roadmap_generator import (
    CONTEXT_JOBS_LIMIT,
    ROADMAP_PROMPT_TEMPLATE,
    SkillGapRoadmap,
    _fallback_roadmap,
    build_prompt,
    generate_roadmap,
    parse_roadmap_json,
    reset_llm_cache,
)

VALID_JSON = (
    '{"skill_gaps": ["Kubernetes", "Terraform"], '
    '"learning_steps": ["Learn Kubernetes", "Learn Terraform"], '
    '"estimated_timeline_weeks": {"Phase 1 - Foundations": "2-3 weeks"}}'
)


def _match(
    *,
    title: str = "Backend Engineer",
    company: str = "Acme",
    similarity: float = 0.85,
    excerpt: str = "Requires Python, Django and PostgreSQL experience.",
) -> MatchResult:
    return MatchResult(
        id="id-1",
        title=title,
        company_display_name=company,
        location_display="London",
        similarity=similarity,
        description_excerpt=excerpt,
    )


class _FakeClient:
    """Minimal duck-typed LLM: returns a fixed string from .invoke()."""

    def __init__(self, output: str | Exception) -> None:
        self._output = output
        self.invoked_text: str | None = None

    def invoke(self, text: str) -> str:
        self.invoked_text = text
        if isinstance(self._output, Exception):
            raise self._output
        return self._output


# --------------------------------------------------------------------------- #
# build_prompt
# --------------------------------------------------------------------------- #


class BuildPromptTests(unittest.TestCase):
    def test_includes_skills(self) -> None:
        prompt = build_prompt(["Python", "Django"], [_match()])
        self.assertIn("Python", prompt)
        self.assertIn("Django", prompt)

    def test_includes_matched_job_details(self) -> None:
        prompt = build_prompt(
            ["Python"],
            [_match(title="Backend Engineer", company="Acme", excerpt="Needs SQL.")],
        )
        self.assertIn("Backend Engineer", prompt)
        self.assertIn("Acme", prompt)
        self.assertIn("Needs SQL.", prompt)

    def test_uses_template(self) -> None:
        prompt = build_prompt(["Python"], [])
        self.assertTrue(prompt.startswith(ROADMAP_PROMPT_TEMPLATE[:30]))

    def test_empty_inputs(self) -> None:
        prompt = build_prompt([], [])
        self.assertIn("(none)", prompt)
        self.assertIn("(no matched jobs available)", prompt)

    def test_respects_context_jobs_limit(self) -> None:
        matches = [_match(title=f"Role {i}", company="") for i in range(10)]
        prompt = build_prompt(["Python"], matches)
        # Only the first CONTEXT_JOBS_LIMIT job titles should appear.
        for i in range(CONTEXT_JOBS_LIMIT):
            self.assertIn(f"Role {i}", prompt)
        self.assertNotIn(f"Role {CONTEXT_JOBS_LIMIT}", prompt)


# --------------------------------------------------------------------------- #
# parse_roadmap_json
# --------------------------------------------------------------------------- #


class ParseRoadmapJsonTests(unittest.TestCase):
    def test_parses_plain_json(self) -> None:
        data = parse_roadmap_json(VALID_JSON)
        self.assertIsInstance(data, dict)
        self.assertIn("skill_gaps", data)

    def test_parses_json_in_code_fences(self) -> None:
        fenced = f"```json\n{VALID_JSON}\n```"
        data = parse_roadmap_json(fenced)
        self.assertIsNotNone(data)
        self.assertEqual(data["skill_gaps"], ["Kubernetes", "Terraform"])

    def test_parses_json_embedded_in_prose(self) -> None:
        prose = f"Here is the roadmap:\n{VALID_JSON}\nHope that helps!"
        data = parse_roadmap_json(prose)
        self.assertIsNotNone(data)
        self.assertIn("learning_steps", data)

    def test_returns_none_for_garbage(self) -> None:
        self.assertIsNone(parse_roadmap_json("no json here at all"))
        self.assertIsNone(parse_roadmap_json(""))

    def test_returns_none_for_valid_non_object_json(self) -> None:
        self.assertIsNone(parse_roadmap_json("[1, 2, 3]"))
        self.assertIsNone(parse_roadmap_json('"just a string"'))


# --------------------------------------------------------------------------- #
# generate_roadmap (fake client)
# --------------------------------------------------------------------------- #


class GenerateRoadmapTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_llm_cache()

    def tearDown(self) -> None:
        reset_llm_cache()

    def test_generated_status_with_valid_output(self) -> None:
        client = _FakeClient(VALID_JSON)
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "generated")
        self.assertEqual(roadmap.skill_gaps, ["Kubernetes", "Terraform"])
        self.assertEqual(roadmap.learning_steps, ["Learn Kubernetes", "Learn Terraform"])
        self.assertEqual(
            roadmap.estimated_timeline_weeks["Phase 1 - Foundations"], "2-3 weeks"
        )

    def test_provenance_fields_are_filled(self) -> None:
        client = _FakeClient(VALID_JSON)
        matches = [_match(title="Backend Engineer"), _match(title="Data Engineer")]
        roadmap = generate_roadmap(["Python"], matches, client=client)
        self.assertEqual(roadmap.matched_jobs, ["Backend Engineer", "Data Engineer"])
        self.assertEqual(roadmap.extracted_skill_count, 1)

    def test_prompt_is_passed_to_client(self) -> None:
        client = _FakeClient(VALID_JSON)
        generate_roadmap(["Python"], [_match()], client=client)
        self.assertIn("Python", client.invoked_text)

    def test_unparseable_output_falls_back(self) -> None:
        client = _FakeClient("totally not json")
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "fallback")
        # Fallback still exposes the schema keys.
        self.assertIsInstance(roadmap.skill_gaps, list)
        self.assertIsInstance(roadmap.learning_steps, list)
        self.assertIsInstance(roadmap.estimated_timeline_weeks, dict)

    def test_invalid_schema_falls_back(self) -> None:
        # skill_gaps must be a list; here it's a string -> ValidationError.
        client = _FakeClient('{"skill_gaps": "Kubernetes"}')
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "fallback")

    def test_client_raise_falls_back(self) -> None:
        client = _FakeClient(RuntimeError("ollama connection refused"))
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "fallback")

    def test_client_factory_raise_falls_back(self) -> None:
        # _get_llm itself blows up (e.g. langchain_ollama missing).
        with mock.patch(
            "api.services.roadmap_generator._get_llm",
            side_effect=RuntimeError("no ollama"),
        ):
            roadmap = generate_roadmap(["Python"], [_match()])
        self.assertEqual(roadmap.status, "fallback")

    def test_empty_input_never_raises(self) -> None:
        client = _FakeClient(VALID_JSON)
        roadmap = generate_roadmap([], [], client=client)
        self.assertEqual(roadmap.status, "generated")
        self.assertEqual(roadmap.skill_gaps, ["Kubernetes", "Terraform"])


# --------------------------------------------------------------------------- #
# Fallback roadmap
# --------------------------------------------------------------------------- #


class FallbackTests(unittest.TestCase):
    def test_detects_gaps_from_matched_specs(self) -> None:
        matches = [
            _match(excerpt="This Kubernetes role needs Terraform and Docker."),
        ]
        roadmap = _fallback_roadmap(
            ["Python", "Docker"], matches, reason="test"
        )
        # Docker is already a candidate skill -> not a gap.
        self.assertNotIn("Docker", roadmap.skill_gaps)
        self.assertIn("Kubernetes", roadmap.skill_gaps)
        self.assertIn("Terraform", roadmap.skill_gaps)

    def test_fallback_has_steps_and_timeline(self) -> None:
        roadmap = _fallback_roadmap(["Python"], [], reason="test")
        self.assertGreater(len(roadmap.learning_steps), 0)
        self.assertGreater(len(roadmap.estimated_timeline_weeks), 0)
        self.assertEqual(roadmap.status, "fallback")

    def test_fallback_empty_input(self) -> None:
        roadmap = _fallback_roadmap([], [], reason="test")
        self.assertEqual(roadmap.status, "fallback")
        self.assertIsInstance(roadmap.skill_gaps, list)
        self.assertIsInstance(roadmap.learning_steps, list)


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #


class SkillGapRoadmapTests(unittest.TestCase):
    def test_to_dict_round_trip(self) -> None:
        r = SkillGapRoadmap(
            status="generated",
            skill_gaps=["Kubernetes"],
            learning_steps=["Learn Kubernetes"],
            estimated_timeline_weeks={"Phase 1": "2-3 weeks"},
            matched_jobs=["Backend Engineer"],
            extracted_skill_count=3,
        )
        d = r.to_dict()
        self.assertEqual(d["status"], "generated")
        self.assertEqual(d["skill_gaps"], ["Kubernetes"])
        self.assertEqual(d["matched_jobs"], ["Backend Engineer"])
        self.assertEqual(d["extracted_skill_count"], 3)

    def test_defaults(self) -> None:
        r = SkillGapRoadmap()
        self.assertEqual(r.status, "generated")
        self.assertEqual(r.skill_gaps, [])
        self.assertEqual(r.learning_steps, [])
        self.assertEqual(r.estimated_timeline_weeks, {})
        self.assertEqual(r.matched_jobs, [])
        self.assertEqual(r.extracted_skill_count, 0)


if __name__ == "__main__":
    unittest.main()
