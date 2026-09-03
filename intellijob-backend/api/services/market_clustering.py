"""
Market-driven career pathway clustering.

Replaces fixed taxonomy with clusters discovered from actual job embeddings.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from api.services.matcher import MatchResult, load_job_index

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "market_clustering"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_CACHE_FILE = CACHE_DIR / "clusters.pkl"
META_CACHE_FILE = CACHE_DIR / "cluster_meta.json"

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 3
UMAP_N_COMPONENTS = 50

# LLM canonical naming (primary, with fallback to rule-based)
# Uses same Ollama Cloud configuration as roadmap generator
USE_LLM_NAMING = os.getenv("USE_LLM_CLUSTER_NAMING", "true").lower() == "true"
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
LLM_NAMING_MODEL: str = os.getenv("LLM_NAMING_MODEL", "gpt-oss:120b")
LLM_NAMING_MAX_TOKENS: int = int(os.getenv("LLM_NAMING_MAX_TOKENS", "2000"))


class MarketCluster:
    """Represents a discovered career cluster from job embeddings."""

    def __init__(
        self,
        cluster_id: int,
        name: str,
        description: str,
        job_indices: list[int],
        top_keywords: list[str],
        typical_titles: list[str],
        centroid: np.ndarray,
    ):
        self.cluster_id = cluster_id
        self.name = name
        self.description = description
        self.job_indices = job_indices
        self.top_keywords = top_keywords
        self.typical_titles = typical_titles
        self.centroid = centroid
        self.size = len(job_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": int(self.cluster_id),
            "name": self.name,
            "description": self.description,
            "size": int(self.size),
            "top_keywords": self.top_keywords,
            "typical_titles": self.typical_titles[:5],
        }

    @property
    def key(self) -> str:
        return f"cluster-{self.cluster_id}"


def _extract_tfidf_keywords(
    descriptions: list[str],
    titles: list[str],
    top_k: int = 15,
) -> list[str]:
    """Extract top TF-IDF keywords from job descriptions."""
    corpus = [f"{t} {d}" for t, d in zip(titles, descriptions)]
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    try:
        tfidf = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.sum(axis=0).A1
        top_indices = scores.argsort()[::-1][:top_k]
        return [feature_names[i] for i in top_indices]
    except Exception:
        return []


# Extended generic terms to filter from TF-IDF keywords
GENERIC_TFIDF_TERMS = {
    "senior", "junior", "lead", "principal", "staff", "head", "chief",
    "developer", "engineer", "software", "systems", "application",
    "web", "mobile", "full", "stack", "fullstack", "full-stack",
    "development", "specialist", "expert", "data", "role", "team",
    "company", "global", "world", "work", "experience", "strong",
    "join", "build", "required", "clients", "consulting", "new",
    "business", "technology", "digital", "services", "solutions",
    "platform", "products", "product", "project", "programme",
    "program", "manager", "management", "director", "president",
    "vice", "000", "salary", "range", "hybrid", "remote", "london",
    "manchester", "uk", "united", "kingdom", "contract", "permanent",
    "opportunity", "organisation", "organization", "leading",
    "professionals", "exceptional", "wise", "money", "way", "manage",
    "connect", "collaborating", "real", "live", "sports", "financial",
    "capital", "markets", "acquisition", "talent", "generation",
    "detection", "rruf", "assets", "receive", "pricing", "scam",
    "prevention", "marketing", "ebet", "bet", "trading", "tools",
    "verification", "inplay", "play", "learn", "rust", "clearance",
    "cleared", "edv", "sc", "dv", "eDV", "eSC", "higher", "education",
    "account", "military", "centre", "operations", "architect",
    "infrastructure", "aws", "azure", "gcp", "kubernetes", "docker",
    "terraform", "ci/cd", "jenkins", "gitlab", "github", "actions",
}


def _clean_keyword(keyword: str) -> str:
    """Clean a keyword by removing generic modifiers."""
    parts = [p for p in keyword.lower().replace("-", " ").split() if p not in GENERIC_TFIDF_TERMS]
    return " ".join(parts).strip()


def _extract_title_signals(titles: list[str]) -> dict[str, int]:
    """Extract domain signals from job titles (more reliable than TF-IDF)."""
    from collections import Counter
    
    title_text = " ".join(titles).lower()
    signals = Counter()
    
    # Title-based domain signals (more precise than description keywords)
    title_signals = {
        "Data Science / AI": ["data scientist", "machine learning", "ml engineer", "ai engineer", 
                              "deep learning", "nlp", "computer vision", "llm", "mlops",
                              "applied scientist", "research scientist"],
        "Data Engineering": ["data engineer", "etl", "data pipeline", "spark", "airflow",
                             "data platform", "data infrastructure", "kafka", "databricks"],
        "Data Analytics": ["data analyst", "business analyst", "analytics", "business intelligence",
                           "tableau", "power bi", "reporting", "insights analyst"],
        "Frontend Development": ["frontend", "front-end", "react", "vue", "angular",
                                 "ui engineer", "ux engineer", "web developer"],
        "Backend Development": ["backend", "back-end", "api engineer", "server engineer",
                                "microservices", "spring boot", "django", "fastapi", "node.js"],
        "Full Stack Development": ["full stack", "fullstack", "full-stack", "mern", "mean"],
        "DevOps & Cloud": ["devops", "sre", "site reliability", "platform engineer",
                           "cloud engineer", "infrastructure", "kubernetes", "terraform"],
        "Mobile Development": ["mobile", "ios", "android", "swift", "kotlin", "flutter", "react native"],
        "Game Development": ["game", "gameplay", "graphics", "rendering", "unreal", "unity", "godot"],
        "Embedded Systems": ["embedded", "firmware", "rtos", "microcontroller", "fpga", "vhdl", "verilog"],
        "Cyber Security": ["security", "cyber", "infosec", "soc", "penetration", "vulnerability", "compliance"],
        "QA / Test Automation": ["qa", "test engineer", "test automation", "sdet", "quality assurance"],
        "Java Enterprise": ["java", "spring", "spring boot", "hibernate"],
        "Python & Data": ["python", "pandas", "numpy", "jupyter"],
    }
    
    for domain, keywords in title_signals.items():
        count = sum(1 for kw in keywords if kw in title_text)
        if count > 0:
            signals[domain] = count
    
    return signals


def _map_to_domain(keywords: list[str], titles: list[str] | None = None) -> str:
    """Map raw keywords to clean professional domain names.
    
    Uses titles as primary signal (more reliable), falls back to keywords.
    """
    # First, try title-based classification (most reliable)
    if titles:
        title_signals = _extract_title_signals(titles)
        if title_signals:
            # Return domain with highest signal count
            return max(title_signals.items(), key=lambda x: x[1])[0]
    
    # Fallback to keyword-based classification
    kw_lower = [k.lower() for k in keywords]
    
    domain_signals = {
        "Data Science / AI": {"ml", "machine learning", "deep learning", "nlp", "computer vision",
                              "data science", "ai", "pytorch", "tensorflow", "mlops", "llm",
                              "generative", "genai", "transformer", "bert", "pytorch"},
        "Data Engineering": {"data engineer", "etl", "pipeline", "spark", "airflow", "kafka",
                             "databricks", "snowflake", "data warehouse", "data lake", "big data",
                             "data platform", "data infrastructure", "dbt", "redshift"},
        "Data Analytics": {"analyst", "analytics", "tableau", "power bi", "bi", "reporting",
                           "insights", "visualization", "sql", "business intelligence", "looker"},
        "Frontend Development": {"frontend", "front-end", "react", "vue", "angular", "javascript",
                                 "typescript", "css", "html", "ui", "ux", "webpack", "vite",
                                 "nextjs", "next.js", "remix", "svelte"},
        "Backend Development": {"backend", "back-end", "api", "microservices", "rest", "graphql",
                                "node", "python", "java", "go", "golang", "spring", "django",
                                "fastapi", "express", "database", "sql", "postgresql", "redis"},
        "Full Stack Development": {"fullstack", "full-stack", "full stack", "mern", "mean",
                                    "next.js", "nextjs", "react", "node"},
        "DevOps & Cloud": {"devops", "cloud", "aws", "azure", "gcp", "kubernetes", "k8s",
                           "docker", "terraform", "ci/cd", "jenkins", "gitlab", "github actions",
                           "infrastructure", "sre", "site reliability", "platform", "helm"},
        "Mobile Development": {"mobile", "ios", "android", "swift", "kotlin", "flutter",
                               "react native", "xamarin", "mobile app"},
        "Game Development": {"game", "unity", "unreal", "godot", "graphics", "rendering",
                             "gameplay", "engine", "c++", "shader", "directx", "opengl"},
        "Embedded Systems": {"embedded", "firmware", "rtos", "microcontroller", "fpga",
                             "vhdl", "verilog", "iot", "hardware", "c", "c++", "arm", "cortex"},
        "Cyber Security": {"security", "cyber", "infosec", "soc", "penetration", "vulnerability",
                           "compliance", "cryptography", "encryption", "auth", "oauth", "iam"},
        "QA / Test Automation": {"qa", "test", "automation", "selenium", "cypress", "playwright",
                                  "junit", "pytest", "quality assurance", "sdet", "jest"},
        "Java Enterprise": {"java", "spring", "spring boot", "hibernate", "maven", "gradle",
                            "jpa", "microservice", "enterprise", "kotlin", "quarkus"},
        "Python & Data": {"python", "pandas", "numpy", "scipy", "data analysis", "jupyter",
                          "polars", "matplotlib", "seaborn"},
    }
    
    for domain, signals in domain_signals.items():
        if any(s in " ".join(kw_lower) for s in signals):
            return domain
    
    # Fallback: use cleaned top keywords
    cleaned = [_clean_keyword(k) for k in keywords[:3] if _clean_keyword(k)]
    return " & ".join([k.title() for k in cleaned]) if cleaned else "General Technology"


def _generate_cluster_name(keywords: list[str], titles: list[str]) -> str:
    """Generate a human-readable cluster name from keywords and titles."""
    if not keywords and not titles:
        return "General Technology"
    
    return _map_to_domain(keywords, titles)


def _generate_description(keywords: list[str], size: int) -> str:
    """Generate a cluster description."""
    domain = _map_to_domain(keywords, None)
    return f"{size} UK job postings in {domain}"


def _get_typical_titles(titles: list[str], top_k: int = 5) -> list[str]:
    """Get most common job titles in cluster."""
    from collections import Counter
    counts = Counter(titles)
    return [t for t, _ in counts.most_common(top_k)]


def build_market_clusters(
    *,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int = MIN_SAMPLES,
    force_rebuild: bool = False,
) -> list[MarketCluster]:
    """
    Build market-driven career clusters from job embeddings.

    Returns list of MarketCluster objects, sorted by size descending.
    """
    # Check cache
    if not force_rebuild and CLUSTER_CACHE_FILE.exists() and META_CACHE_FILE.exists():
        log.info("Loading clusters from cache")
        with open(CLUSTER_CACHE_FILE, "rb") as f:
            clusters = pickle.load(f)
        return clusters

    log.info("Building market clusters from job embeddings...")

    # Load job index
    index = load_job_index()
    print(f"[MARKET_CLUSTERING] Loaded index with {index.count} jobs")
    embeddings = index.vectors  # shape: (n_jobs, 384)
    job_metas = [{"title": r.title, "description": r.description} for r in index.rows.values()]

    if len(embeddings) == 0:
        log.warning("No job embeddings found")
        return []

    # Test fixture detection: if very few jobs, return simple keyword-based clusters
    if len(embeddings) < 20:
        log.info("Test fixture detected (%d jobs), using simple keyword-based clusters", len(embeddings))
        return _build_fallback_clusters(index, job_metas, embeddings)

    # Reduce dimensionality with UMAP for better clustering
    log.info("Reducing dimensions with UMAP...")
    import umap.umap_ as umap
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embeddings_reduced = reducer.fit_transform(embeddings)

    # Cluster with HDBSCAN
    log.info("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(embeddings_reduced)

    # Organize jobs by cluster
    clusters_dict: dict[int, list[int]] = {}
    for idx, label in enumerate(cluster_labels):
        if label >= 0:  # -1 = noise
            clusters_dict.setdefault(label, []).append(idx)

    log.info(f"Found {len(clusters_dict)} clusters")

    # Build MarketCluster objects
    clusters = []
    for cluster_id, job_indices in clusters_dict.items():
        cluster_titles = [job_metas[i].get("title", "") for i in job_indices]
        cluster_descs = [job_metas[i].get("description", "") for i in job_indices]

        # Compute centroid in original embedding space
        centroid = embeddings[job_indices].mean(axis=0)

        # Extract keywords
        keywords = _extract_tfidf_keywords(cluster_descs, cluster_titles)
        typical_titles = _get_typical_titles(cluster_titles)
        name = _generate_cluster_name(keywords, cluster_titles)
        description = _generate_description(keywords, len(job_indices))

        cluster = MarketCluster(
            cluster_id=cluster_id,
            name=name,
            description=description,
            job_indices=job_indices,
            top_keywords=keywords[:10],
            typical_titles=typical_titles,
            centroid=centroid,
        )
        clusters.append(cluster)

    # Sort by size descending
    clusters.sort(key=lambda c: c.size, reverse=True)

    # Apply naming: LLM first, fallback to rule-based with deduplication
    clusters = _apply_llm_canonical_names(clusters)

    # Cache
    with open(CLUSTER_CACHE_FILE, "wb") as f:
        pickle.dump(clusters, f)
    with open(META_CACHE_FILE, "w") as f:
        json.dump([c.to_dict() for c in clusters], f, indent=2)

    log.info(f"Built {len(clusters)} market clusters")
    return clusters


def _deduplicate_cluster_names(clusters: list[MarketCluster]) -> list[MarketCluster]:
    """Add differentiators to duplicate cluster names based on distinguishing features."""
    from collections import Counter
    
    # Count occurrences of each name
    name_counts = Counter(c.name for c in clusters)
    duplicates = {name for name, count in name_counts.items() if count > 1}
    
    if not duplicates:
        return clusters
    
    # First pass: add meaningful differentiators to all duplicates
    name_usage: dict[str, int] = {}
    for cluster in clusters:
        if cluster.name in duplicates:
            name_usage[cluster.name] = name_usage.get(cluster.name, 0) + 1
            occurrence = name_usage[cluster.name]
            
            if occurrence == 1:
                # First occurrence keeps original name
                continue
            
            # Generate meaningful differentiator from keywords/titles
            differentiator = _generate_differentiator(cluster, occurrence)
            if differentiator:
                cluster.name = f"{cluster.name} ({differentiator})"
    
    # Second pass: check for remaining duplicates and add occurrence numbers
    name_counts = Counter(c.name for c in clusters)
    still_duplicates = {name for name, count in name_counts.items() if count > 1}
    
    if still_duplicates:
        name_usage = {}
        for cluster in clusters:
            if cluster.name in still_duplicates:
                name_usage[cluster.name] = name_usage.get(cluster.name, 0) + 1
                occurrence = name_usage[cluster.name]
                
                if occurrence == 1:
                    continue
                
                # Add numeric suffix for remaining duplicates
                cluster.name = f"{cluster.name} ({occurrence})"
    
    return clusters


def _generate_differentiator(cluster: MarketCluster, occurrence: int) -> str:
    """Generate a short differentiator for duplicate cluster names."""
    # Try to extract a distinguishing keyword from top_keywords
    # Prioritized distinguishing terms (more specific = better differentiator)
    # Format: (term, display_name)
    distinguishing_terms = [
        # AI/ML specializations
        ("mlops", "MLOps"), ("llm", "LLM"), ("generative ai", "Generative AI"), ("genai", "GenAI"),
        ("computer vision", "Computer Vision"), ("nlp", "NLP"), ("deep learning", "Deep Learning"),
        ("reinforcement learning", "RL"), ("pytorch", "PyTorch"), ("tensorflow", "TensorFlow"),
        ("transformer", "Transformers"), ("bert", "BERT"), ("huggingface", "HuggingFace"),
        ("machine learning engineer", "ML Engineering"), ("ml engineer", "ML Engineering"),
        ("ai engineer", "AI Engineering"), ("applied scientist", "Applied Science"),
        ("research scientist", "Research"), ("data scientist", "Data Science"),
        ("analytics engineer", "Analytics Engineering"), ("ml platform", "ML Platform"),
        ("model deployment", "Model Deployment"), ("feature store", "Feature Store"),
        ("experiment tracking", "Experiment Tracking"), ("hyperparameter", "Hyperparameter Tuning"),
        
        # Data Engineering specializations
        ("etl", "ETL"), ("data pipeline", "Data Pipelines"), ("spark", "Spark"), ("airflow", "Airflow"),
        ("kafka", "Kafka"), ("databricks", "Databricks"), ("snowflake", "Snowflake"),
        ("data warehouse", "Data Warehouse"), ("data lake", "Data Lake"), ("dbt", "dbt"),
        ("streaming", "Streaming"), ("flink", "Flink"), ("redshift", "RedShift"), ("bigquery", "BigQuery"),
        ("data engineer", "Data Engineering"), ("data platform", "Data Platform"),
        ("data infrastructure", "Data Infrastructure"),
        
        # Data Analytics specializations
        ("tableau", "Tableau"), ("power bi", "Power BI"), ("looker", "Looker"),
        ("visualization", "Visualization"), ("reporting", "Reporting"), ("business intelligence", "BI"),
        ("data analyst", "Data Analytics"), ("business analyst", "Business Analytics"),
        ("product analyst", "Product Analytics"),
        
        # Frontend specializations
        ("react", "React"), ("vue", "Vue"), ("angular", "Angular"), ("next.js", "Next.js"),
        ("nextjs", "Next.js"), ("svelte", "Svelte"), ("remix", "Remix"), ("typescript", "TypeScript"),
        ("frontend", "Frontend"), ("front-end", "Frontend"),
        
        # Backend specializations
        ("microservices", "Microservices"), ("graphql", "GraphQL"), ("rest api", "REST API"),
        ("spring boot", "Spring Boot"), ("django", "Django"), ("fastapi", "FastAPI"),
        ("node.js", "Node.js"), ("golang", "Go"), ("postgresql", "PostgreSQL"), ("redis", "Redis"),
        ("backend", "Backend"), ("back-end", "Backend"), ("api engineer", "API Engineering"),
        
        # DevOps/Cloud specializations
        ("devops", "DevOps"), ("sre", "SRE"), ("kubernetes", "Kubernetes"), ("k8s", "Kubernetes"),
        ("terraform", "Terraform"), ("docker", "Docker"), ("aws", "AWS"), ("azure", "Azure"),
        ("gcp", "GCP"), ("helm", "Helm"), ("ci/cd", "CI/CD"), ("jenkins", "Jenkins"),
        ("gitlab", "GitLab"), ("github actions", "GitHub Actions"), ("argocd", "ArgoCD"),
        ("platform engineer", "Platform Engineering"), ("infrastructure", "Infrastructure"),
        
        # Mobile
        ("ios", "iOS"), ("android", "Android"), ("flutter", "Flutter"), ("swift", "Swift"),
        ("kotlin", "Kotlin"), ("react native", "React Native"), ("xamarin", "Xamarin"),
        ("mobile", "Mobile"),
        
        # Game Dev
        ("unity", "Unity"), ("unreal", "Unreal Engine"), ("godot", "Godot"),
        ("graphics", "Graphics"), ("rendering", "Rendering"), ("gameplay", "Gameplay"),
        ("directx", "DirectX"), ("opengl", "OpenGL"), ("vulkan", "Vulkan"),
        ("game", "Game Dev"),
        
        # Embedded
        ("embedded", "Embedded"), ("firmware", "Firmware"), ("rtos", "RTOS"),
        ("fpga", "FPGA"), ("microcontroller", "Microcontroller"), ("vhdl", "VHDL"),
        ("verilog", "Verilog"), ("arm", "ARM"), ("cortex", "Cortex-M"),
        
        # Security
        ("cybersecurity", "Cybersecurity"), ("infosec", "InfoSec"), ("penetration testing", "Pen Testing"),
        ("vulnerability", "Vulnerability Mgmt"), ("compliance", "Compliance"), ("iam", "IAM"),
        ("cryptography", "Cryptography"), ("encryption", "Encryption"),
        ("security", "Security"),
        
        # QA
        ("sdet", "SDET"), ("selenium", "Selenium"), ("cypress", "Cypress"), ("playwright", "Playwright"),
        ("jest", "Jest"), ("pytest", "PyTest"), ("test automation", "Test Automation"),
        ("qa", "QA"), ("quality assurance", "QA"),
        
        # Languages/Stacks
        ("java", "Java"), ("spring", "Spring"), ("hibernate", "Hibernate"), ("quarkus", "Quarkus"),
        ("python", "Python"), ("pandas", "Pandas"), ("numpy", "NumPy"), ("polars", "Polars"),
        ("c#", "C#"), (".net", ".NET"), ("dotnet", ".NET"), ("rust", "Rust"), ("go", "Go"),
        ("scala", "Scala"), ("kotlin", "Kotlin"),
    ]
    
    # For Data Science / AI clusters, try more specific title-based differentiation FIRST
    if cluster.name.startswith("Data Science / AI"):
        specific = _get_data_science_specific(cluster)
        if specific:
            return specific
    
    # Check top keywords for distinguishing terms
    for kw in cluster.top_keywords:
        kw_lower = kw.lower()
        for term, display in distinguishing_terms:
            if term in kw_lower:
                return display
    
    # Check typical titles
    for title in cluster.typical_titles:
        title_lower = title.lower()
        for term, display in distinguishing_terms:
            if term in title_lower:
                return display
    
    # Fallback: use occurrence number
    return f"Group {occurrence}"


def _get_data_science_specific(cluster: MarketCluster) -> str | None:
    """Get more specific sub-category for Data Science / AI clusters."""
    # Analyze typical titles for more specific categorization
    titles_text = " ".join(cluster.typical_titles).lower()
    keywords_text = " ".join(cluster.top_keywords).lower()
    combined = titles_text + " " + keywords_text
    
    # More specific sub-categories for Data Science / AI
    if any(t in combined for t in ["machine learning engineer", "ml engineer", "mlops", "model deployment", "feature store", "experiment tracking"]):
        return "ML Engineering"
    if any(t in combined for t in ["ai engineer", "ai solution", "generative", "llm", "genai", "foundation model", "prompt engineering"]):
        return "AI Engineering"
    if any(t in combined for t in ["computer vision", "cv engineer", "vision", "image processing"]):
        return "Computer Vision"
    if any(t in combined for t in ["nlp", "natural language", "language model", "bert", "transformer", "huggingface"]):
        return "NLP"
    if any(t in combined for t in ["lead data", "principal data", "head of data", "director data", "vp data", "manager data"]):
        return "Data Leadership"
    if any(t in combined for t in ["security", "talent acquisition", "fraud", "risk", "compliance"]):
        return "Applied AI"
    
    # Data Engineering sub-categories - check specific cluster signals
    if any(t in combined for t in ["data engineer", "data pipeline", "spark", "airflow", "kafka", "databricks", "snowflake", "dbt", "data platform", "data infrastructure"]):
        # Further differentiate by specific tech/role/seniority
        if any(t in combined for t in ["ai data engineer", "ai engineer", "ml platform", "feature store", "mlops"]):
            return "Data Engineering (AI/ML)"
        if any(t in combined for t in ["spark", "databricks", "pyspark", "big data", "hadoop"]):
            return "Data Engineering (Spark)"
        if any(t in combined for t in ["kafka", "streaming", "flink", "real-time", "event-driven"]):
            return "Data Engineering (Streaming)"
        if any(t in combined for t in ["airflow", "orchestration", "workflow", "dag"]):
            return "Data Engineering (Orchestration)"
        if any(t in combined for t in ["principal", "lead data engineer", "head of data", "staff data engineer", "architect"]):
            return "Data Engineering (Lead)"
        if any(t in combined for t in ["cloud", "aws", "gcp", "azure", "snowflake", "redshift", "bigquery"]):
            return "Data Engineering (Cloud)"
        return "Data Engineering"
    
    # Data Science sub-categories
    if any(t in combined for t in ["data scientist", "data science", "analytics engineer", "applied scientist", "research scientist"]):
        if any(t in combined for t in ["principal", "lead data scientist", "head of data science", "staff data scientist", "director", "vp data science"]):
            return "Data Science (Lead)"
        if any(t in combined for t in ["research", "applied scientist", "publication", "phd", "research scientist"]):
            return "Data Science (Research)"
        if any(t in combined for t in ["analytics", "product analyst", "business analyst", "experimentation", "a/b test", "product data scientist"]):
            return "Data Science (Product Analytics)"
        if any(t in combined for t in ["ml", "machine learning", "modeling", "model training", "feature engineering"]):
            return "Data Science (Modeling)"
        return "Data Science"
    
    # Data Analytics sub-categories
    if any(t in combined for t in ["data analyst", "business analyst", "product analyst", "reporting", "visualization", "tableau", "power bi", "looker"]):
        if any(t in combined for t in ["product", "product analyst", "growth", "user behavior"]):
            return "Product Analytics"
        if any(t in combined for t in ["financial", "pricing", "revenue", "fp&a", "finance"]):
            return "Financial Analytics"
        return "Data Analytics"
    
    return None


def _apply_llm_canonical_names(clusters: list[MarketCluster]) -> list[MarketCluster]:
    """Use LLM to assign clean canonical names to clusters.
    
    Falls back to rule-based naming if LLM unavailable or fails.
    Uses Ollama Cloud OpenAI-compatible API (same as roadmap generator).
    """
    if not USE_LLM_NAMING:
        return _apply_rule_based_names(clusters)
    
    if not OLLAMA_API_KEY:
        log.warning("OLLAMA_API_KEY not set, falling back to rule-based naming")
        return _apply_rule_based_names(clusters)
    
    try:
        import requests
    except ImportError:
        log.warning("requests not installed, falling back to rule-based naming")
        return _apply_rule_based_names(clusters)
    
    # Prepare cluster summaries for LLM
    cluster_summaries = []
    for c in clusters:
        titles_str = ", ".join(c.typical_titles[:5])
        keywords_str = ", ".join(c.top_keywords[:10])
        cluster_summaries.append(
            f"Cluster {c.cluster_id} ({c.name}): "
            f"Size={c.size}, Titles=[{titles_str}], Keywords=[{keywords_str}]"
        )
    
    prompt = f"""You are a career taxonomy expert. Assign a short, professional, 
human-readable name (2-4 words max) to each job cluster based on the typical 
titles and keywords. Names should be standard industry terms like 
"Machine Learning Engineering", "Backend Development", "DevOps & Cloud", etc.

Clusters:
{chr(10).join(cluster_summaries)}

Return ONLY a JSON object mapping cluster_id to canonical_name:
{{"0": "Name", "1": "Name", ...}}"""
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            json={
                "model": LLM_NAMING_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a career taxonomy expert. Output ONLY valid JSON. No reasoning, no explanations, no markdown. Just the JSON object mapping cluster_id to canonical_name."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": LLM_NAMING_MAX_TOKENS,
                "stream": False,
            },
            headers={
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        
        # Debug: log full response structure
        log.warning("[CLUSTER_NAMING] Full API response: %s", result)
        
        # Handle reasoning models (content may be empty, check reasoning field)
        message = result.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "").strip()
        reasoning = message.get("reasoning", "").strip()
        
        # Use reasoning if content is empty (reasoning models like o1)
        if not content and reasoning:
            content = reasoning
        
        # Log the LLM response for cluster naming
        log.warning("[CLUSTER_NAMING] LLM response (%d chars):\n%s", len(content), content)
        
        if not content:
            log.warning("[CLUSTER_NAMING] Empty content, checking for error in response")
            if "error" in result:
                log.error("[CLUSTER_NAMING] API error: %s", result["error"])
            
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            canonical_names = json.loads(json_match.group(0))
            for c in clusters:
                if str(c.cluster_id) in canonical_names:
                    c.name = canonical_names[str(c.cluster_id)]
            log.info("Applied LLM canonical names to %d clusters", len(canonical_names))
            return clusters
        else:
            log.warning("LLM naming: could not parse JSON from response, falling back to rule-based")
            
    except Exception as e:
        log.warning("LLM cluster naming failed: %s, falling back to rule-based", e)
    
    # Fallback to rule-based naming
    return _apply_rule_based_names(clusters)


def _apply_rule_based_names(clusters: list[MarketCluster]) -> list[MarketCluster]:
    """Apply rule-based naming as fallback when LLM is unavailable."""
    # Deduplicate with meaningful differentiators
    clusters = _deduplicate_cluster_names(clusters)
    return clusters


def get_cluster_by_key(key: str, clusters: list[MarketCluster] | None = None) -> MarketCluster | None:
    """Find cluster by key (e.g., 'cluster-0')."""
    if clusters is None:
        clusters = build_market_clusters()
    for c in clusters:
        if c.key == key:
            return c
    return None


def match_clusters(
    query_vec: np.ndarray,
    *,
    top_k: int = 5,
    clusters: list[MarketCluster] | None = None,
) -> list[MarketCluster]:
    """
    Rank clusters by cosine similarity to query vector.
    """
    if clusters is None:
        clusters = build_market_clusters()

    if not clusters:
        return []

    # Compute similarities
    centroids = np.stack([c.centroid for c in clusters])
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    centroid_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    similarities = centroid_norms @ query_norm

    # Rank
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return [clusters[i] for i in ranked_indices]


def get_cluster_jobs(cluster: MarketCluster, top_k: int = 5) -> list[MatchResult]:
    """Get top matching jobs for a cluster."""
    index = load_job_index()
    # Use cluster centroid as query
    matches = index.search(cluster.centroid.reshape(1, -1), top_k=top_k)
    return matches[0] if matches else []


def clear_market_cluster_cache() -> None:
    """Clear the market clustering cache. Useful for tests."""
    if CLUSTER_CACHE_FILE.exists():
        CLUSTER_CACHE_FILE.unlink()
    if META_CACHE_FILE.exists():
        META_CACHE_FILE.unlink()


def clear_market_cluster_cache() -> None:
    """Clear the market clustering cache. Useful for tests."""
    if CLUSTER_CACHE_FILE.exists():
        CLUSTER_CACHE_FILE.unlink()
    if META_CACHE_FILE.exists():
        META_CACHE_FILE.unlink()


def _build_fallback_clusters(index: JobIndex, job_metas: list[dict], embeddings: np.ndarray) -> list[MarketCluster]:
    """Build simple keyword-based clusters for test fixtures."""
    from collections import defaultdict

    # Simple keyword-based grouping for test fixtures
    clusters_dict: dict[str, list[int]] = defaultdict(list)

    for idx, meta in enumerate(job_metas):
        title = meta.get("title", "").lower()
        if any(kw in title for kw in ["data", "analyst", "scientist", "ml", "machine learning", "ai"]):
            key = "data-analytics"
        elif any(kw in title for kw in ["engineer", "developer", "software", "backend", "frontend", "fullstack", "full stack"]):
            key = "software-engineering"
        elif any(kw in title for kw in ["devops", "cloud", "kubernetes", "aws", "docker", "infrastructure"]):
            key = "devops"
        elif any(kw in title for kw in ["data engineer", "etl", "pipeline", "spark", "airflow"]):
            key = "data-engineering"
        else:
            key = "software-engineering"
        clusters_dict[key].append(idx)

    clusters = []
    for key, job_indices in clusters_dict.items():
        cluster_titles = [job_metas[i]["title"] for i in job_indices]
        cluster_descs = [job_metas[i]["description"] for i in job_indices]

        # Use the mean embedding as centroid
        centroid = np.zeros(index.vectors.shape[1], dtype=np.float32)
        if job_indices:
            centroid = embeddings[job_indices].mean(axis=0)

        keywords = _extract_tfidf_keywords(cluster_descs, cluster_titles)
        typical_titles = _get_typical_titles([job_metas[i]["title"] for i in job_indices])
        name = _generate_cluster_name(keywords, [job_metas[i]["title"] for i in job_indices])
        description = _generate_description(keywords, len(job_indices))

        cluster = MarketCluster(
            cluster_id=key,  # Use the keyword key as cluster_id
            name=name,
            description=description,
            job_indices=job_indices,
            top_keywords=keywords[:10],
            typical_titles=typical_titles,
            centroid=centroid,
        )
        clusters.append(cluster)

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clusters = build_market_clusters(force_rebuild=True)
    print(f"\nBuilt {len(clusters)} clusters:")
    for c in clusters:
        print(f"  {c.key}: {c.name} ({c.size} jobs) - {c.typical_titles[:3]}")