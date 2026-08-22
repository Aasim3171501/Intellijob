"""
Tests for api.services.skill_extractor.

Strategy: mock pymupdf4llm.to_markdown so tests don't depend on a
real PDF (and don't ship a binary fixture). The EntityRuler still
runs against the real spaCy en_core_web_sm pipeline — that gives us
real coverage of "do our SKILL_PATTERNS actually match real text".

For tests that only need the text-to-skills path, we never call
the PDF layer at all. The PDF-path test mocks pymupdf4llm.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from api.services.skill_extractor import (
    DEFAULT_SKILL_LABEL,
    GAZETTEER_PATH,
    SKILL_PATTERNS,
    extract_skills_from_pdf,
    extract_skills_from_text,
    extract_text_from_pdf,
    get_nlp,
    pattern_surface,
)


def _fresh_nlp():
    """Force a clean nlp pipeline so tests can't bleed cached state."""
    get_nlp.cache_clear()
    return get_nlp()


class GazetteerTests(unittest.TestCase):
    def test_patterns_are_non_empty(self) -> None:
        self.assertGreater(len(SKILL_PATTERNS), 10)

    def test_all_patterns_have_skill_label(self) -> None:
        bad = [p for p in SKILL_PATTERNS if p.get("label") != DEFAULT_SKILL_LABEL]
        self.assertEqual(bad, [], f"Non-SKILL patterns found: {bad}")

    def test_patterns_are_dicts_with_pattern_key(self) -> None:
        bad = [p for p in SKILL_PATTERNS if "pattern" not in p]
        self.assertEqual(bad, [])

    def test_every_pattern_has_valid_token_specs(self) -> None:
        # Each token spec must be a single-key dict using only the
        # attributes EntityRuler accepts for our gazetteer.
        for p in SKILL_PATTERNS:
            self.assertGreaterEqual(len(p["pattern"]), 1)
            for spec in p["pattern"]:
                self.assertEqual(
                    len(spec), 1,
                    f"Pattern {p!r} has multi-key token spec: {spec}",
                )
                key = next(iter(spec))
                self.assertIn(
                    key, ("LOWER", "TEXT"),
                    f"Pattern {p!r} uses unsupported attribute {key!r}",
                )

    def test_no_duplicate_surface_forms(self) -> None:
        seen: set[str] = set()
        for p in SKILL_PATTERNS:
            surface = pattern_surface(p).lower()
            self.assertNotIn(surface, seen, f"Duplicate skill pattern: {surface}")
            seen.add(surface)

    def test_gazetteer_file_exists_and_is_valid(self) -> None:
        self.assertTrue(GAZETTEER_PATH.exists())
        with open(GAZETTEER_PATH, encoding="utf-8") as fh:
            gazetteer = json.load(fh)
        for section in ("case_insensitive", "case_sensitive", "phrases"):
            self.assertIn(section, gazetteer)
        total = (
            sum(len(v) for v in gazetteer["case_insensitive"].values())
            + sum(len(v) for v in gazetteer["case_sensitive"].values())
            + len(gazetteer["phrases"])
        )
        self.assertGreater(total, 10)

    def test_common_word_terms_are_case_sensitive(self) -> None:
        # Terms that are ordinary English words must NOT be matched as
        # single tokens case-insensitively ("go" would hit every sentence).
        # (They may still appear inside phrases: "spring boot", "ruby on rails".)
        gazetteer = json.loads(GAZETTEER_PATH.read_text(encoding="utf-8"))
        for term in ("go", "react", "vue", "spring", "swift", "ruby", "r", "chef", "puppet", "lean", "express"):
            in_lower = any(
                term in skills
                for skills in gazetteer["case_insensitive"].values()
            )
            self.assertFalse(
                in_lower,
                f"{term!r} is an English word but is matched case-insensitively",
            )

    def test_case_sensitive_terms_are_not_duplicated_in_lower(self) -> None:
        gazetteer = json.loads(GAZETTEER_PATH.read_text(encoding="utf-8"))
        lower = {
            skill.lower()
            for skills in gazetteer["case_insensitive"].values()
            for skill in skills
        }
        for skills in gazetteer["case_sensitive"].values():
            for skill in skills:
                self.assertNotIn(
                    skill.lower(), lower,
                    f"{skill!r} appears in both case_sensitive and "
                    f"case_insensitive sections",
                )


class SkillExtractionTests(unittest.TestCase):
    """Hits the real spaCy pipeline. First call pays the model-load
    cost (~0.3s); subsequent tests reuse the cached nlp object."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.nlp = _fresh_nlp()
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(
                f"spaCy model en_core_web_sm not loadable: {e}"
            )

    def test_extracts_known_skill(self) -> None:
        text = "Built backends in Python with Django and PostgreSQL."
        skills = extract_skills_from_text(text)
        self.assertIn("Python", skills)
        self.assertIn("Django", skills)
        self.assertIn("PostgreSQL", skills)

    def test_extracts_multiple_skills_sorted(self) -> None:
        text = "Experience with Kubernetes, Docker, AWS, and Terraform."
        skills = extract_skills_from_text(text)
        # Sorted alphabetically (case-insensitive)
        self.assertEqual(skills, ["AWS", "Docker", "Kubernetes", "Terraform"])

    def test_returns_empty_list_for_empty_text(self) -> None:
        self.assertEqual(extract_skills_from_text(""), [])
        self.assertEqual(extract_skills_from_text("   \n\t  "), [])

    def test_returns_empty_list_when_no_skills(self) -> None:
        skills = extract_skills_from_text(
            "The quick brown fox jumps over the lazy dog."
        )
        self.assertEqual(skills, [])

    def test_deduplicates_case_insensitive(self) -> None:
        text = "python PYTHON Python PyThOn developer."
        skills = extract_skills_from_text(text)
        # All four mentions match the LOWER pattern and collapse to the
        # first-seen casing.
        self.assertEqual(skills, ["python"])

    def test_matches_case_insensitively(self) -> None:
        text = "built backends in PYTHON with django and postgresql."
        skills = extract_skills_from_text(text)
        self.assertIn("PYTHON", skills)
        self.assertIn("django", skills)
        self.assertIn("postgresql", skills)

    def test_common_word_skills_require_exact_casing(self) -> None:
        # "react" / "go" are ordinary English words — lowercase must NOT
        # be flagged as skills, only the capitalised framework/language.
        lowercase = extract_skills_from_text("how to react to change and go home")
        self.assertEqual(lowercase, [])
        capitalised = extract_skills_from_text("I use React and Go daily.")
        self.assertIn("React", capitalised)
        self.assertIn("Go", capitalised)

    def test_extracts_multi_token_phrases(self) -> None:
        text = "Built machine learning models and a REST API."
        skills = extract_skills_from_text(text)
        self.assertIn("machine learning", skills)
        self.assertIn("REST API", skills)

    def test_extracts_symbol_skills(self) -> None:
        text = "C# and scikit-learn are in my toolkit."
        skills = extract_skills_from_text(text)
        self.assertIn("C#", skills)
        self.assertIn("scikit-learn", skills)

    def test_extracts_newly_added_skills(self) -> None:
        # Terms that came in with the JSON gazetteer, not the old list.
        text = "Kafka, Spark, Jenkins, GraphQL, and Terraform."
        skills = extract_skills_from_text(text)
        for tech in ("Kafka", "Spark", "Jenkins", "GraphQL", "Terraform"):
            self.assertIn(tech, skills, f"Missing newly added skill {tech}")

    def test_strips_trailing_punctuation(self) -> None:
        text = "Proficient in Python, Django, PostgreSQL."
        skills = extract_skills_from_text(text)
        self.assertIn("Python", skills)
        self.assertIn("Django", skills)
        # PostgreSQL had a trailing period in the source — should be cleaned.
        self.assertTrue(
            all(not s.endswith(".") for s in skills),
            f"Found skill with trailing punctuation: {skills}",
        )

    def test_skill_label_takes_precedence_over_org(self) -> None:
        # spaCy's default NER labels PostgreSQL as ORG sometimes.
        # With our ruler inserted before NER and overwrite_ents=True,
        # SKILL should win.
        text = "I administer PostgreSQL clusters."
        skills = extract_skills_from_text(text)
        self.assertIn("PostgreSQL", skills)

    def test_real_world_cv_style_text(self) -> None:
        text = """
        Senior Backend Engineer with 6 years experience.
        Stack: Python, Django, FastAPI, PostgreSQL, Redis, Docker,
        Kubernetes, AWS, Terraform, Git, Agile.
        """
        skills = extract_skills_from_text(text)
        # Every one of these should be present.
        expected = {
            "Python", "Django", "FastAPI", "PostgreSQL", "Redis",
            "Docker", "Kubernetes", "AWS", "Terraform", "Git", "Agile",
        }
        missing = expected - set(skills)
        self.assertEqual(missing, set(), f"Missing skills: {missing}")


class PdfExtractionTests(unittest.TestCase):
    """Mock pymupdf4llm.to_markdown so we don't ship a PDF fixture."""

    def test_extract_text_calls_pymupdf4llm(self) -> None:
        fake_pdf = b"%PDF-1.4 fake bytes"
        with mock.patch(
            "pymupdf4llm.to_markdown", return_value="# Header\n\nPython dev."
        ) as m:
            out = extract_text_from_pdf(fake_pdf)
        self.assertEqual(out, "# Header\n\nPython dev.")
        m.assert_called_once()
        # pymupdf4llm 1.28 accepts a path string (not BytesIO / bytes).
        # The arg should be a str that ends in .pdf.
        args, _kwargs = m.call_args
        self.assertIsInstance(args[0], str)
        self.assertTrue(args[0].endswith(".pdf"))

    def test_extract_skills_from_pdf_end_to_end(self) -> None:
        fake_pdf = b"%PDF-1.4 fake"
        with mock.patch(
            "pymupdf4llm.to_markdown",
            return_value="Built apps in Python and Django.",
        ):
            skills = extract_skills_from_pdf(fake_pdf)
        self.assertIn("Python", skills)
        self.assertIn("Django", skills)

    def test_extract_text_from_pdf_uses_tempfile(self) -> None:
        """The tempfile should be unlinked after the call returns."""
        import os
        import tempfile as _tempfile

        captured_paths: list[str] = []

        original_mkstemp = _tempfile.mkstemp

        def _capturing_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            captured_paths.append(path)
            return fd, path

        with mock.patch("api.services.skill_extractor.tempfile.mkstemp",
                        side_effect=_capturing_mkstemp), \
             mock.patch("pymupdf4llm.to_markdown",
                        return_value="hello"):
            extract_text_from_pdf(b"%PDF-1.4 fake")

        self.assertEqual(len(captured_paths), 1)
        # File must be cleaned up after the call.
        self.assertFalse(
            os.path.exists(captured_paths[0]),
            f"Tempfile not cleaned up: {captured_paths[0]}",
        )


@unittest.skip("Opt-in slow test: exercises every gazetteer entry against real spaCy")
class GazetteerExhaustivenessTests(unittest.TestCase):
    def test_every_pattern_finds_itself_in_a_sentence(self) -> None:
        _fresh_nlp()
        for p in SKILL_PATTERNS:
            # Multi-token patterns (phrases, C#, scikit-learn) need a
            # specially constructed sentence; only single-token patterns
            # can be self-tested with a plain "I have experience with X".
            if len(p["pattern"]) != 1:
                continue
            surface = next(iter(p["pattern"][0].values()))
            text = f"I have experience with {surface}."
            skills = extract_skills_from_text(text)
            self.assertIn(
                surface, skills,
                f"Gazetteer pattern {surface!r} not detected in plain sentence",
            )


if __name__ == "__main__":
    unittest.main()