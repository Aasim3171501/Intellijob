"""
Tests for api.services.roadmap_generator.

Strategy:

* The prompt builder is pure string code — tested directly.
* JSON parsing is a pure function — tested with plain JSON, fenced
  JSON, and prose-embedded JSON.
* generate_roadmap is tested with an injected fake ``client`` object
  (duck-typed ``.invoke(text) -> str``), so no network is ever touched
  and the tests stay fast and deterministic.
* Fallback paths are exercised for: unparseable output, invalid schema
  output, client exceptions, and an unavailable client factory.
* Skill-gap fallback detection uses a real MatchResult fixture whose
  description excerpt names technologies from the gazetteer.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from api.services.matcher import MatchResult
from api.services.roadmap_generator import (
    CONTEXT_JOBS_LIMIT,
    MAX_OUTPUT_TOKENS,
    ROADMAP_PROMPT_TEMPLATE,
    TEMPERATURE,
    LearningStep,
    SkillGapRoadmap,
    _fallback_roadmap,
    build_prompt,
    generate_roadmap,
    parse_roadmap_json,
    reset_llm_cache,
)

VALID_JSON = (
    '{"skill_gaps": ["Kubernetes", "Terraform"], '
    '"learning_steps": ['
    '{"id": "step-1", "title": "Learn Kubernetes", '
    '"overview": "Master pods and deployments.", '
    '"primary_skill": "Kubernetes", "estimated_hours": 20}, '
    '{"id": "step-2", "title": "Learn Terraform", '
    '"overview": "Write IaC for a real stack.", '
    '"primary_skill": "Terraform", "estimated_hours": 15}], '
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

    def test_parses_fenced_json_with_trailing_prose_on_same_line(self) -> None:
        # Closing fence followed by prose on the same line, where the prose
        # itself contains braces — the old rfind("}") parser would have
        # swallowed the trailing text and failed to load.
        fenced = f"```json\n{VALID_JSON}\n``` Hope {{that}} helps!"
        data = parse_roadmap_json(fenced)
        self.assertIsNotNone(data)
        self.assertEqual(data["skill_gaps"], ["Kubernetes", "Terraform"])

    def test_parses_first_complete_object_when_prose_has_braces(self) -> None:
        data = parse_roadmap_json(f"text {{not json}} then {VALID_JSON} end")
        self.assertIsNotNone(data)
        self.assertEqual(data["learning_steps"][0]["id"], "step-1")

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
        self.assertEqual(len(roadmap.learning_steps), 2)
        first = roadmap.learning_steps[0]
        self.assertEqual(first.id, "step-1")
        self.assertEqual(first.title, "Learn Kubernetes")
        self.assertEqual(first.overview, "Master pods and deployments.")
        self.assertEqual(first.primary_skill, "Kubernetes")
        self.assertEqual(first.estimated_hours, 20)
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
        # A bare array can't be a roadmap object -> parse fails -> fallback.
        client = _FakeClient("[1, 2, 3]")
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "fallback")

    def test_partial_schema_is_sanitized_not_fallback(self) -> None:
        # A good generation with one sloppy field (skill_gaps as a single
        # string) must be coerced, not rejected into fallback.
        client = _FakeClient('{"skill_gaps": "Kubernetes"}')
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "generated")
        self.assertEqual(roadmap.skill_gaps, ["Kubernetes"])

    def test_alternative_schema_is_normalized_not_fallback(self) -> None:
        # gpt-oss:120b sometimes ignores the requested schema and returns a
        # different key set (role/learning_path/milestones/key_skills/...).
        # Such output must be mapped onto the roadmap schema, not discarded.
        client = _FakeClient(
            '{"role": "Backend Engineer", '
            '"learning_path": [{"id": "step-1", "title": "Learn FastAPI", '
            '"overview": "Build a REST API.", "primary_skill": "FastAPI", '
            '"estimated_hours": 20}], '
            '"key_skills": ["FastAPI", "PostgreSQL"], '
            '"tools_and_technologies": ["Docker"], '
            '"career_progression": [{"name": "Phase 1", "focus": "Foundations", '
            '"objectives": ["Ship a project"]}]}'
        )
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "generated")
        self.assertIn("FastAPI", roadmap.skill_gaps)
        self.assertIn("Docker", roadmap.skill_gaps)
        self.assertEqual(roadmap.learning_steps[0].title, "Learn FastAPI")
        self.assertEqual(roadmap.career_trajectory[0].name, "Phase 1")

    def test_trailing_comma_json_is_repaired(self) -> None:
        # A common LLM defect: trailing commas before closing braces/brackets.
        # Without repair the top-level object fails to parse and the model's
        # good generation would be discarded into fallback.
        client = _FakeClient(
            '{"skill_gaps": ["Kubernetes",], '
            '"learning_steps": [{"id": "step-1", "title": "Learn Kubernetes", '
            '"overview": "Master pods.", "primary_skill": "Kubernetes", '
            '"estimated_hours": 20},], '
            '"estimated_timeline_weeks": {"Phase 1 - Foundations": "2-3 weeks",}}'
        )
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "generated")
        self.assertEqual(roadmap.skill_gaps, ["Kubernetes"])
        self.assertEqual(len(roadmap.learning_steps), 1)

    def test_wrapped_roadmap_is_unwrapped(self) -> None:
        # The model occasionally nests the roadmap inside a wrapper object.
        client = _FakeClient(
            '{"roadmap": ' + VALID_JSON + "}"
        )
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "generated")
        self.assertEqual(roadmap.skill_gaps, ["Kubernetes", "Terraform"])

    def test_empty_roadmap_from_model_falls_back(self) -> None:
        # Even valid JSON with nothing in it must not reach the UI as an
        # empty "No Roadmap Available" — fall back to the deterministic one.
        client = _FakeClient(
            '{"skill_gaps": [], "learning_steps": [], '
            '"estimated_timeline_weeks": {}, "career_trajectory": []}'
        )
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "fallback")
        self.assertGreater(len(roadmap.learning_steps), 0)

    def test_zero_hour_step_does_not_trigger_fallback(self) -> None:
        # The LLM occasionally emits estimated_hours: 0; that must be
        # clamped, not reject the whole roadmap into fallback.
        client = _FakeClient(
            '{"skill_gaps": ["Kubernetes"], '
            '"learning_steps": [{"id": "step-1", "title": "Learn Kubernetes", '
            '"overview": "Master pods.", "primary_skill": "Kubernetes", '
            '"estimated_hours": 0}], '
            '"estimated_timeline_weeks": {"Phase 1": "2-3 weeks"}}'
        )
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "generated")
        self.assertEqual(roadmap.learning_steps[0].estimated_hours, 1)

    def test_client_raise_falls_back(self) -> None:
        client = _FakeClient(RuntimeError("llm api error"))
        roadmap = generate_roadmap(["Python"], [_match()], client=client)
        self.assertEqual(roadmap.status, "fallback")

    def test_no_client_uses_deterministic_fallback(self) -> None:
        # With no LLM provider configured, generate_roadmap returns the
        # deterministic roadmap instead of raising or returning nothing.
        with mock.patch.dict(os.environ, {}, clear=True):
            reset_llm_cache()
            roadmap = generate_roadmap(["Python"], [_match()])
        self.assertEqual(roadmap.status, "fallback")
        self.assertIsInstance(roadmap.learning_steps, list)
        self.assertGreater(len(roadmap.learning_steps), 0)

    def test_no_api_key_uses_fallback(self) -> None:
        from api.services import roadmap_generator as rg

        reset_llm_cache()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(rg._get_llm_client())
        # End-to-end: no OLLAMA_API_KEY -> deterministic fallback, no raise.
        reset_llm_cache()
        with mock.patch.dict(os.environ, {}, clear=True):
            roadmap = generate_roadmap(["Python"], [_match()])
        self.assertEqual(roadmap.status, "fallback")
        self.assertIsInstance(roadmap.learning_steps, list)

    def test_ollama_adapter_sends_chat_completion_payload(self) -> None:
        from api.services import roadmap_generator as rg

        captured: dict = {}

        class _FakeResponse:
            def __init__(self, data: dict) -> None:
                self._data = data

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self._data

        class _FakeSession:
            def post(self, url, json, headers, timeout):
                captured["url"] = url
                captured["payload"] = json
                captured["headers"] = headers
                captured["timeout"] = timeout
                return _FakeResponse(
                    {
                        "choices": [
                            {"message": {"content": VALID_JSON}}
                        ]
                    }
                )

        adapter = rg._OllamaRoadmapClient(
            api_key="test-key",
            model="gpt-oss:120b",
            session=_FakeSession(),
        )
        roadmap = adapter.invoke("hello")
        self.assertIsInstance(roadmap, rg.SkillGapRoadmap)
        self.assertEqual(roadmap.skill_gaps, ["Kubernetes", "Terraform"])
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["payload"]["model"], "gpt-oss:120b")
        self.assertEqual(captured["payload"]["messages"][0]["role"], "user")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "hello")
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["temperature"], TEMPERATURE)
        self.assertEqual(captured["payload"]["max_tokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(
            captured["headers"]["Authorization"], "Bearer test-key"
        )

    def test_ollama_adapter_http_error_falls_back(self) -> None:
        from api.services import roadmap_generator as rg

        class _FakeResponse:
            def raise_for_status(self) -> None:
                raise RuntimeError("401 Unauthorized")

            def json(self) -> dict:
                raise AssertionError("should not be called")

        class _FakeSession:
            def post(self, url, json, headers, timeout):
                return _FakeResponse()

        adapter = rg._OllamaRoadmapClient(
            api_key="bad-key",
            model="gpt-oss:120b",
            session=_FakeSession(),
        )
        roadmap = generate_roadmap(["Python"], [_match()], client=adapter)
        self.assertEqual(roadmap.status, "fallback")

    def test_slow_llm_times_out_and_falls_back(self) -> None:
        # A stalled model call must never hang the request: the invoke runs
        # in a worker thread bounded by LLM_TIMEOUT_SECONDS and falls back
        # deterministically.
        import time

        from api.services import roadmap_generator as rg

        class _SlowClient:
            def invoke(self, text: str) -> str:
                time.sleep(0.3)
                return VALID_JSON

        started = time.monotonic()
        with mock.patch.object(rg, "LLM_TIMEOUT_SECONDS", 0.05):
            roadmap = generate_roadmap(["Python"], [_match()], client=_SlowClient())
        elapsed = time.monotonic() - started

        self.assertEqual(roadmap.status, "fallback")
        # Returns quickly (well under the slow client's sleep) — proving
        # the abandoned generation didn't block the response.
        self.assertLess(elapsed, 0.25)

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
        # (Gap names come from the gazetteer, lower-cased.)
        gaps_lower = [g.lower() for g in roadmap.skill_gaps]
        self.assertNotIn("docker", gaps_lower)
        self.assertIn("kubernetes", gaps_lower)
        self.assertIn("terraform", gaps_lower)

    def test_fallback_has_steps_and_timeline(self) -> None:
        roadmap = _fallback_roadmap(["Python"], [], reason="test")
        self.assertGreater(len(roadmap.learning_steps), 0)
        self.assertGreater(len(roadmap.estimated_timeline_weeks), 0)
        self.assertEqual(roadmap.status, "fallback")
        # Every fallback step adheres to the LearningStep schema.
        for step in roadmap.learning_steps:
            self.assertIsInstance(step, LearningStep)
            self.assertTrue(step.id)
            self.assertTrue(step.title)
            self.assertTrue(step.overview)
            self.assertTrue(step.primary_skill)
            self.assertGreaterEqual(step.estimated_hours, 1)

    def test_fallback_steps_are_uniquely_identified(self) -> None:
        roadmap = _fallback_roadmap(
            ["Python", "Docker"],
            [_match(excerpt="This Kubernetes role needs Terraform.")],
            reason="test",
        )
        ids = [s.id for s in roadmap.learning_steps]
        self.assertEqual(len(ids), len(set(ids)), f"Duplicate step ids: {ids}")

    def test_fallback_empty_input(self) -> None:
        roadmap = _fallback_roadmap([], [], reason="test")
        self.assertEqual(roadmap.status, "fallback")
        self.assertIsInstance(roadmap.skill_gaps, list)
        self.assertIsInstance(roadmap.learning_steps, list)


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #


class LearningStepTests(unittest.TestCase):
    def test_zero_hours_clamped_to_one(self) -> None:
        step = LearningStep(
            id="step-1",
            title="Learn Kubernetes",
            overview="Master pods.",
            primary_skill="Kubernetes",
            estimated_hours=0,
        )
        self.assertEqual(step.estimated_hours, 1)

    def test_negative_hours_clamped_to_one(self) -> None:
        step = LearningStep(
            id="step-1",
            title="Learn Kubernetes",
            overview="Master pods.",
            primary_skill="Kubernetes",
            estimated_hours=-5,
        )
        self.assertEqual(step.estimated_hours, 1)

    def test_float_hours_truncated(self) -> None:
        step = LearningStep(
            id="step-1",
            title="Learn Kubernetes",
            overview="Master pods.",
            primary_skill="Kubernetes",
            estimated_hours=2.9,
        )
        self.assertEqual(step.estimated_hours, 2)

    def test_valid_hours_preserved(self) -> None:
        step = LearningStep(
            id="step-1",
            title="Learn Kubernetes",
            overview="Master pods.",
            primary_skill="Kubernetes",
            estimated_hours=20,
        )
        self.assertEqual(step.estimated_hours, 20)


class SkillGapRoadmapTests(unittest.TestCase):
    def test_to_dict_round_trip(self) -> None:
        r = SkillGapRoadmap(
            status="generated",
            skill_gaps=["Kubernetes"],
            learning_steps=[
                LearningStep(
                    id="step-1",
                    title="Learn Kubernetes",
                    overview="Master pods.",
                    primary_skill="Kubernetes",
                    estimated_hours=20,
                )
            ],
            estimated_timeline_weeks={"Phase 1": "2-3 weeks"},
            matched_jobs=["Backend Engineer"],
            extracted_skill_count=3,
        )
        d = r.to_dict()
        self.assertEqual(d["status"], "generated")
        self.assertEqual(d["skill_gaps"], ["Kubernetes"])
        self.assertEqual(d["learning_steps"][0]["title"], "Learn Kubernetes")
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
