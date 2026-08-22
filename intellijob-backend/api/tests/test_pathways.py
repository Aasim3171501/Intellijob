"""
Tests for api.services.pathways.

Strategy:

* Reuse the tiny npz + CSV fixture pattern from test_matcher: no model,
  no real data, sha256-seeded unit vectors.
* classify() is pure string logic — tested directly with a spread of
  real-world-ish titles.
* build_pathway_index / match_pathways / match_jobs_in_pathway are
  exercised over a fixture whose titles span several pathways, so
  centroids, member assignments and ranking are all checked against
  reproducible data.
"""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from api.services.embedder import EMBEDDING_DIM
from api.services.jobs_loader import CSV_COLUMNS
from api.services.matcher import load_job_index, reset_job_index_cache
from api.services.pathways import (
    CAREER_PATHWAYS,
    FALLBACK_PATHWAY_KEY,
    Pathway,
    PathwayIndex,
    PathwayMatch,
    build_pathway_index,
    classify,
    match_jobs_in_pathway,
    match_pathways,
    pathway_by_key,
)

#: (title, expected pathway key) pairs for classify().
CLASSIFY_CASES: tuple[tuple[str, str], ...] = (
    ("Embedded Software Engineer", "embedded"),
    ("Firmware Engineer", "embedded"),
    ("Senior Data Scientist", "ml-ai"),
    ("Machine Learning Engineer", "ml-ai"),
    ("Data Engineer", "data-eng"),
    ("ETL Developer", "data-eng"),
    ("Data Analyst", "data-analytics"),
    ("Business Intelligence Analyst", "data-analytics"),
    ("Frontend Developer", "frontend"),
    ("Backend Engineer", "backend"),
    ("Full Stack Developer", "fullstack"),
    ("DevOps Engineer", "devops"),
    ("Site Reliability Engineer", "devops"),
    ("QA Automation Engineer", "qa"),
    ("Security Engineer", "security"),
    ("Solution Architect", "architecture"),
    ("Android Developer", "mobile"),
    ("Gameplay Programmer", "game"),
    ("Software Engineer", "software-engineering"),
    ("Senior Software Engineer", "software-engineering"),
    # Language-titled roles classify to backend, not the catch-all.
    ("Python Developer", "backend"),
    ("Golang Developer", "backend"),
    ("C++ Developer", "backend"),
    # "programme" is a substring of "programmer" — a plain "programmer"
    # title must NOT be swallowed by the delivery pathway.
    ("Frontend Programmer", "frontend"),
    ("Python Programmer", "backend"),
    # Delivery variants.
    ("Program Manager", "delivery"),
    ("Programme Manager", "delivery"),
    # Embedded/hardware variants.
    ("Control Systems Engineer", "embedded"),
    # Game variants.
    ("Graphics Programmer", "game"),
    ("", "software-engineering"),
)


def _sha_unit_vec(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


realistic_titles: list[str] = [
    "Embedded Software Engineer",
    "Embedded Systems Engineer",
    "Data Scientist",
    "Data Analyst",
    "DevOps Engineer",
    "Backend Engineer",
    "Software Engineer",
    "Senior Software Engineer",
    "Frontend Developer",
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)


class _DiverseFixture:
    """Tiny index whose titles span several pathways."""

    def __init__(self) -> None:
        self.titles = list(realistic_titles)
        self.tmp = Path(tempfile.mkdtemp(prefix="hermes-pathways-"))
        self.npz = self.tmp / "jobs.npz"
        self.csv = self.tmp / "jobs.csv"

        ids = [f"id-{i}" for i in range(len(self.titles))]
        vectors = np.stack([_sha_unit_vec(t) for t in self.titles], axis=0).astype(np.float32)
        np.savez_compressed(
            self.npz,
            ids=np.asarray(ids, dtype=object),
            vectors=vectors,
            model_name=np.asarray("fake-model"),
            schema_version=np.asarray(1, dtype=np.int64),
        )
        _write_csv(self.csv, [
            {
                **{c: "" for c in CSV_COLUMNS if c not in ("id", "title", "description")},
                "id": id_,
                "title": title,
                "description": f"Role {i} requires Python and experience.",
            }
            for i, (id_, title) in enumerate(zip(ids, self.titles))
        ])

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


class ClassifyTests(unittest.TestCase):
    def test_maps_titles_to_expected_keys(self) -> None:
        for title, expected in CLASSIFY_CASES:
            self.assertEqual(classify(title), expected, msg=f"title={title!r}")

    def test_all_keys_exist_in_taxonomy(self) -> None:
        known = {p.key for p in CAREER_PATHWAYS}
        for _, key in CLASSIFY_CASES:
            self.assertIn(key, known)

    def test_pathway_by_key(self) -> None:
        self.assertIsInstance(pathway_by_key("embedded"), Pathway)
        self.assertIsNone(pathway_by_key("nope"))

    def test_fallback_exists_as_keywordless_entry(self) -> None:
        fallback = pathway_by_key(FALLBACK_PATHWAY_KEY)
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.keywords, ())


class PathwayIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _DiverseFixture()
        reset_job_index_cache()
        self.idx = load_job_index(str(self.fx.npz), str(self.fx.csv))

    def tearDown(self) -> None:
        self.fx.cleanup()
        reset_job_index_cache()

    def test_builds_centroids_and_members(self) -> None:
        pi = build_pathway_index(self.idx)
        self.assertIsInstance(pi, PathwayIndex)
        self.assertEqual(pi.dimension, EMBEDDING_DIM)

        # Two embedded titles -> 2 members.
        self.assertEqual(len(pi.member_positions["embedded"]), 2)
        # software-engineering catches the generic titles.
        self.assertIn("software-engineering", pi.member_positions)
        self.assertGreater(len(pi.member_positions["software-engineering"]), 0)
        # Every pathway key in the taxonomy has an entry.
        for p in CAREER_PATHWAYS:
            self.assertIn(p.key, pi.centroids)
            self.assertIn(p.key, pi.member_positions)

    def test_centroids_are_unit_norm_when_non_empty(self) -> None:
        pi = build_pathway_index(self.idx)
        for p in CAREER_PATHWAYS:
            c = pi.centroids[p.key]
            self.assertEqual(c.shape, (EMBEDDING_DIM,))
            if pi.member_positions[p.key]:
                self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=5)
            else:
                # Empty pathways have a zero centroid.
                self.assertEqual(float(np.linalg.norm(c)), 0.0)

    def test_pathway_index_total_members_equals_index_count(self) -> None:
        pi = build_pathway_index(self.idx)
        total = sum(len(v) for v in pi.member_positions.values())
        self.assertEqual(total, self.idx.count)


class MatchPathwaysTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _DiverseFixture()
        reset_job_index_cache()
        self.idx = load_job_index(str(self.fx.npz), str(self.fx.csv))
        self.pi = build_pathway_index(self.idx)

    def tearDown(self) -> None:
        self.fx.cleanup()
        reset_job_index_cache()

    def test_top_pathway_is_cosine_max(self) -> None:
        # Query aligned exactly with the 'embedded' centroid -> it wins.
        q = self.pi.centroids["embedded"].copy()
        ranked = match_pathways(q, index=self.idx, pathway_index=self.pi)
        self.assertEqual(ranked[0].key, "embedded")
        self.assertAlmostEqual(ranked[0].similarity, 1.0, places=5)

    def test_top_k_respected_and_sorted(self) -> None:
        q = self.pi.centroids["backend"].copy()
        ranked = match_pathways(q, top_k=3, index=self.idx, pathway_index=self.pi)
        self.assertEqual(len(ranked), 3)
        for r in ranked:
            self.assertIsInstance(r, PathwayMatch)
            self.assertTrue(r.name)
            self.assertGreaterEqual(r.match_count, 0)
        sims = [r.similarity for r in ranked]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_empty_query_still_returns_top_k(self) -> None:
        q = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        ranked = match_pathways(q, top_k=3, index=self.idx, pathway_index=self.pi)
        self.assertEqual(len(ranked), 3)

    def test_zero_top_k_returns_empty(self) -> None:
        q = self.pi.centroids["embedded"].copy()
        self.assertEqual(match_pathways(q, top_k=0, index=self.idx, pathway_index=self.pi), [])

    def test_match_count_matches_member_size(self) -> None:
        q = self.pi.centroids["embedded"].copy()
        ranked = match_pathways(q, top_k=1, index=self.idx, pathway_index=self.pi)
        self.assertEqual(ranked[0].match_count, len(self.pi.member_positions["embedded"]))

    def test_skill_aware_ranking_favours_pathway_matching_skills(self) -> None:
        # Tiny index with two pathways whose descriptions mention disjoint
        # skill sets, and a query perfectly aligned with 'embedded'.
        # Passing skills that only match 'frontend' must overturn the
        # pure-cosine order — this is what makes recommendations
        # resume-specific instead of always the same top-k.
        titles = [
            "Embedded Software Engineer",  # embedded
            "Firmware Engineer",           # embedded
            "Frontend Developer",          # frontend
            "React Developer",             # frontend
        ]
        descriptions = [
            "Firmware, RTOS and microcontroller work.",
            "Firmware, RTOS and microcontroller work.",
            "Build UIs with React and TypeScript.",
            "Build UIs with React and TypeScript.",
        ]
        tmp = Path(tempfile.mkdtemp(prefix="hermes-skills-"))
        try:
            ids = [f"id-{i}" for i in range(len(titles))]
            vectors = np.stack([_sha_unit_vec(t) for t in titles], axis=0).astype(np.float32)
            np.savez_compressed(
                tmp / "jobs.npz",
                ids=np.asarray(ids, dtype=object),
                vectors=vectors,
                model_name=np.asarray("fake-model"),
                schema_version=np.asarray(1, dtype=np.int64),
            )
            _write_csv(tmp / "jobs.csv", [
                {
                    **{c: "" for c in CSV_COLUMNS if c not in ("id", "title", "description")},
                    "id": id_,
                    "title": title,
                    "description": desc,
                }
                for id_, title, desc in zip(ids, titles, descriptions)
            ])
            reset_job_index_cache()
            idx = load_job_index(str(tmp / "jobs.npz"), str(tmp / "jobs.csv"))
            pi = build_pathway_index(idx)

            # Query aligned exactly with the 'embedded' centroid.
            q = pi.centroids["embedded"].copy()
            ranked = match_pathways(
                q,
                top_k=2,
                index=idx,
                pathway_index=pi,
                skills=["React", "TypeScript"],
            )
            self.assertEqual(ranked[0].key, "frontend")
            self.assertEqual(sorted(ranked[0].matched_skills), ["React", "TypeScript"])
            self.assertGreater(ranked[0].coverage, 0)
            self.assertEqual(ranked[1].coverage, 0)
            # The blend must lie within [0, 1] for a human-readable score.
            self.assertGreaterEqual(ranked[0].score, 0)
            self.assertLessEqual(ranked[0].score, 1)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            reset_job_index_cache()

    def test_exclude_fallback_removes_catch_all(self) -> None:
        # A query aligned with the catch-all centroid is #1 by cosine, but
        # discovery mode must not recommend the leftovers bucket.
        q = self.pi.centroids[FALLBACK_PATHWAY_KEY].copy()
        with_fallback = match_pathways(q, top_k=3, index=self.idx, pathway_index=self.pi)
        self.assertEqual(with_fallback[0].key, FALLBACK_PATHWAY_KEY)

        without = match_pathways(
            q,
            top_k=3,
            index=self.idx,
            pathway_index=self.pi,
            exclude_fallback=True,
        )
        self.assertNotIn(FALLBACK_PATHWAY_KEY, {r.key for r in without})
        self.assertEqual(len(without), 3)


class MatchJobsInPathwayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _DiverseFixture()
        reset_job_index_cache()
        self.idx = load_job_index(str(self.fx.npz), str(self.fx.csv))
        self.pi = build_pathway_index(self.idx)

    def tearDown(self) -> None:
        self.fx.cleanup()
        reset_job_index_cache()

    def test_only_members_of_pathway_returned(self) -> None:
        # 'embedded' has exactly 2 member jobs.
        q = self.pi.centroids["embedded"].copy()
        results = match_jobs_in_pathway(q, "embedded", top_k=5, index=self.idx, pathway_index=self.pi)
        self.assertEqual(len(results), 2)
        titles = {r.title for r in results}
        self.assertTrue(any("Embedded" in t for t in titles))
        self.assertNotIn("Data Scientist", titles)

    def test_ranked_by_similarity(self) -> None:
        q = self.pi.centroids["embedded"].copy()
        results = match_jobs_in_pathway(q, "embedded", top_k=5, index=self.idx, pathway_index=self.pi)
        sims = [r.similarity for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertIn("Embedded", results[0].title)

    def test_unknown_key_returns_empty(self) -> None:
        q = self.pi.centroids["embedded"].copy()
        self.assertEqual(match_jobs_in_pathway(q, "nope", top_k=5, index=self.idx, pathway_index=self.pi), [])


class TaxomomySanityTests(unittest.TestCase):
    def test_keys_unique(self) -> None:
        keys = [p.key for p in CAREER_PATHWAYS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_fallback_is_last(self) -> None:
        self.assertEqual(CAREER_PATHWAYS[-1].key, FALLBACK_PATHWAY_KEY)


if __name__ == "__main__":
    unittest.main()