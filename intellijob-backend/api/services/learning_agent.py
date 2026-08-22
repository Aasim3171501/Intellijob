"""
Learning resource agent — searches Tavily for current, high-quality learning links.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Any

import requests

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "learning_resources"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

KIND_BY_DOMAIN = {
    "documentation": {
        "docs.", "developer.", "learn.", "docs.microsoft.com", "developer.hashicorp.com",
        "kubernetes.io", "redis.io", "postgresql.org", "mongodb.com", "scikit-learn.org",
        "fastapi.tiangolo.com", "spring.io", "react.dev", "nodejs.org", "go.dev",
        "docs.docker.com", "docs.aws.amazon.com", "learn.microsoft.com",
    },
    "tutorial": {
        "w3schools.", "freecodecamp.", "baeldung.", "geeksforgeeks.", "tutorial", "guide",
        "realpython.", "python-", "java-", "javascript.", "typescriptlang.",
    },
    "interactive": {
        "roadmap.sh", "codecademy.", "datacamp.", "coursera.", "edx.", "udemy.",
        "pluralsight.", "linkedin.com/learning", "kodekloud.", "katacoda.",
    },
}

SKILL_QUERY_TEMPLATES = [
    "{skill} official documentation tutorial",
    "{skill} learning path roadmap",
    "{skill} best practices guide",
]

def _infer_kind(url: str) -> str:
    url_lower = url.lower()
    for kind, domains in KIND_BY_DOMAIN.items():
        if any(d in url_lower for d in domains):
            return kind
    return "tutorial"

def _cache_path(skill: str) -> Path:
    key = hashlib.sha256(skill.lower().encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"

def _load_cache(skill: str) -> list[dict[str, Any]] | None:
    path = _cache_path(skill)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("ts", 0) < CACHE_TTL_SECONDS:
            return data.get("resources")
    except Exception:
        pass
    return None

def _save_cache(skill: str, resources: list[dict[str, Any]]) -> None:
    path = _cache_path(skill)
    try:
        path.write_text(json.dumps({"ts": time.time(), "resources": resources}))
    except Exception:
        pass

def search_learning_resources(skill: str, max_results: int = 6) -> list[dict[str, Any]]:
    """
    Returns a list of dicts: {title, url, domain, kind}.
    Uses cache; falls back to empty list on any error.
    """
    cached = _load_cache(skill)
    if cached is not None:
        return cached

    if not TAVILY_API_KEY:
        return []

    resources: list[dict[str, Any]] = []
    seen_urls = set()

    for tmpl in SKILL_QUERY_TEMPLATES:
        query = tmpl.format(skill=skill)
        try:
            resp = requests.post(
                TAVILY_ENDPOINT,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": 5,
                    "search_depth": "basic",
                    "include_domains": [],
                    "exclude_domains": ["youtube.com", "youtu.be"],
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("results", []):
                url = item.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = item.get("title") or url
                domain = url.split("/")[2] if len(url.split("/")) > 2 else ""
                kind = _infer_kind(url)
                resources.append({
                    "title": title[:120],
                    "url": url,
                    "domain": domain,
                    "kind": kind,
                })
                if len(resources) >= max_results:
                    break
        except Exception:
            continue
        if len(resources) >= max_results:
            break

    # Ensure we have at least one of each kind if possible
    kinds_present = {r["kind"] for r in resources}
    for kind in ("documentation", "tutorial", "interactive"):
        if kind not in kinds_present and resources:
            # promote one to the missing kind
            resources[0]["kind"] = kind
            kinds_present.add(kind)

    _save_cache(skill, resources)
    return resources