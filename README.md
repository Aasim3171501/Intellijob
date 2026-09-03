# IntelliJob

> A decoupled, offline-first career analytics platform that turns a candidate's PDF résumé into a personalised, AI-generated skill-gap roadmap with strategic career trajectories — built for the MSc IT+ dissertation at the **University of Glasgow**.

IntelliJob shifts away from generic live job-board scrapers. It acts as a **deterministic, strategic career advisor**: it parses résumés, extracts skills, performs sub-50 ms semantic vector search against a UK software-engineering job dataset, and synthesises a context-grounded learning plan via a **local** Retrieval-Augmented Generation (RAG) pipeline. **No candidate data ever leaves the machine.**

---

## ✨ Features

### Core Analysis
- 📄 **Deterministic PDF résumé parsing** via `PyMuPDF` (clean text extraction, layout-aware).
- 🧠 **Custom NLP skill tokenisation** using `spaCy` + a hand-tuned `EntityRuler` for technical entities (`SKILL`).
- 🔢 **384-dim semantic embeddings** with `SentenceTransformers` (`all-MiniLM-L6-v2`).
- ⚡ **Sub-50 ms vector similarity search** via in-memory NumPy cosine similarity (file-based embeddings); PostgreSQL/pgvector/HNSW planned for production scale.

### Dual-Mode Analysis
- 🎯 **Targeted Mode** — User specifies a target role (e.g., "Senior Backend Engineer") → single skill-gap roadmap against matched job specs.
- 🔍 **Discovery Mode** — Target left blank → market-driven career pathway discovery with ranked recommendations, each with its own roadmap and trajectory.

### Market-Driven Career Pathways
- 📊 **HDBSCAN clustering** on job embeddings → discovers career pathways from actual market data (1056 UK Adzuna jobs → 29 clusters).
- 🏷️ **LLM-powered canonical naming** — Ollama Cloud assigns clean, professional names (e.g., "Machine Learning Engineering", "DevOps & Cloud Engineering") with automatic deduplication.
- 📈 **Hybrid ranking** — Cosine similarity to cluster centroids + skill-overlap coverage for personalised recommendations.
- 🔁 **Lazy-loaded roadmaps** — Only pathway #1 generated upfront; others generated on-demand when clicked (prevents LLM request queueing).

### Strategic Career Trajectory
- 🗺️ **3-phase progression** — Entry-Level Readiness → Mid-Level Progression → Senior Trajectory.
- 💰 **Realistic UK salary ranges** per phase.
- 🎯 **Phase-specific objectives**, typical titles, required skills, and unlock criteria.
- 🔄 **PhaseDetailModal** with expandable week-by-week plans + "Retry with AI" on deterministic fallback.

### Learning & Resources
- 📚 **Learning Steps** — Prioritised milestones with time estimates, primary skills, overviews.
- 🌐 **Tavily-powered live resources** — Official docs, tutorials, interactive courses (cached 30 days).
- 🧩 **Expandable StepDetailModal** — Deep-dive per milestone with curated resource links.

### UX & Resilience
- ⚙️ **Loading states** — Spinners, skeleton screens, disabled clicks during LLM generation.
- 🛡️ **Graceful degradation** — Deterministic fallback roadmaps when LLM unavailable; "Retry with AI" button.
- 🔒 **100 % local processing** — No candidate data leaves the machine; GDPR-friendly by design.
- 🧱 **Decoupled architecture** — React SPA + Django REST Framework API (file-based embeddings, in-memory search; PostgreSQL/pgvector planned for production scale).

---

## 🏗️ Architecture

### Current Implementation (Two-Tier + File-Based Storage)

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1 — React SPA (Vite + TS + Tailwind + ShadCN)            │
│            Axios → Django endpoints                             │
└─────────────────────────────────────────────────────────────────┘
                              │  HTTP / JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2 — Django REST Framework (Python 3.11+)                 │
│   PyMuPDF ─▶ spaCy NER ─▶ SentenceTransformers ─▶ Roadmap Gen  │
│   Market Clustering (HDBSCAN/UMAP) ─▶ LLM Naming ─▶ Lazy Roads │
│   In-Memory Vector Search (NumPy cosine similarity)            │
└─────────────────────────────────────────────────────────────────┘
                              │  File I/O (pickle / .npz)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 3 — Local File Cache                                      │
│   job_index.pkl  │  clusters.pkl  │  cluster_meta.json         │
│   Embeddings: .npz files (NumPy arrays, 384-dim)               │
│   Source: 1056 UK Adzuna jobs → sentence-transformers          │
└─────────────────────────────────────────────────────────────────┘
```

### Planned Production Architecture (PostgreSQL + pgvector)

*For multi-user deployments requiring persistence, concurrent access, and horizontal scaling:*

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1 — React SPA (Vite + TS + Tailwind + ShadCN)            │
└─────────────────────────────────────────────────────────────────┘
                              │  HTTP / JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2 — Django REST Framework + Celery Workers               │
│   PyMuPDF ─▶ spaCy NER ─▶ SentenceTransformers ─▶ Roadmap Gen  │
│   Async task queue for LLM calls, clustering, embedding jobs   │
└─────────────────────────────────────────────────────────────────┘
                              │  pgvector SQL
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 3 — PostgreSQL 15+ + pgvector + HNSW index               │
│   Jobs table: id, title, description, embedding (vector), meta │
│   Users table: profiles, saved roadmaps, progress tracking     │
│   HNSW index for sub-50ms semantic search at scale             │
└─────────────────────────────────────────────────────────────────┘
```

### Current Data Flow (File-Based)

### End-to-End Pipeline

#### 1. Offline Preparation (One-Time / Periodic Refresh)
```
Adzuna API → Job Seeding → Embedding (all-MiniLM-L6-v2) 
    → HDBSCAN/UMAP Clustering → LLM Canonical Naming 
    → Cache to disk (clusters.pkl + cluster_meta.json)
```

#### 2. Real-Time User Loop (Per Request)

**Targeted Mode:**
```
PDF Upload → Text Extraction → Skill Extraction (spaCy)
    → Query Vector → In-Memory Cosine Search (Top-k Job Matches)
    → LLM Roadmap Generation → Response
```

**Discovery Mode:**
```
PDF Upload → Text Extraction → Skill Extraction (spaCy)
    → Query Vector → Cosine Similarity to Cluster Centroids
    → Hybrid Score (Embedding + Skill Coverage) → Rank Pathways
    → Filter (Score ≥ 0.30, Coverage ≥ 0.15, Min 3 Max 4)
    → Pathway #1: Generate Roadmap + Matches (Lazy-load others)
    → Response with pathways[], pathway #1 roadmap, trajectory
```

---

## 🗂️ Monorepo Layout

```
intellijob/
├── .gitignore
├── .gitlab-ci.yml          # ESLint + Ruff lint pipeline
├── README.md
├── intellijob-backend/     # Django REST Framework API
│   ├── venv/
│   ├── requirements.txt
│   ├── manage.py
│   ├── core/               # Django project (settings, urls, asgi/wsgi)
│   └── api/                # Application logic
│       ├── services/
│       │   ├── matcher.py              # Embeddings + in-memory cosine search
│       │   ├── skill_extractor.py      # PyMuPDF + spaCy NER
│       │   ├── roadmap_generator.py    # LLM roadmap + phase plans
│       │   ├── market_clustering.py    # HDBSCAN + UMAP + LLM naming
│       │   ├── pathways_market.py      # Market pathway matching
│       │   ├── pathways.py             # Fixed taxonomy (test fallback)
│       │   └── learning_agent.py       # Tavily resource search
│       ├── views.py                    # AnalyzeView, PhasePlanView, PathwayRoadmapView
│       └── tests/                      # Comprehensive test suite
└── intellijob-frontend/    # React + TypeScript SPA (Vite)
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    └── src/
        ├── components/
        │   ├── CareerPathways.tsx     # Pathway cards + lazy loading
        │   ├── RoadmapView.tsx        # Skill gaps, steps, trajectory
        │   ├── PhaseDetailModal.tsx   # 3-phase + week plans + retry
        │   ├── LearningSteps.tsx      # Milestones + resource links
        │   ├── MatchedRoles.tsx       # Role pills
        │   └── SkillPills.tsx         # Extracted skill tags
        ├── App.tsx                    # Main orchestrator + lazy loading logic
        ├── lib/api.ts                 # Axios client + pathway roadmap endpoint
        └── types/api.ts               # TypeScript interfaces
```

---

## 🚀 Getting Started

### Prerequisites

| Tool                | Version           | Notes                                               |
| ------------------- | ----------------- | --------------------------------------------------- |
| Python              | 3.11+             | Backend runtime                                     |
| Node.js             | 20 LTS or newer   | Frontend runtime                                    |
| PostgreSQL          | 15+               | **Planned for production**; not used in current file-based implementation |
| Ollama              | latest            | Local LLM runtime (`ollama pull gpt-oss:120b`)      |

### 1. Clone

```bash
git clone https://stgit.dcs.gla.ac.uk/msc-project-for-information-technology/2025/it-project-3171501k/intellijob.git
cd intellijob
```

### 2. Backend

```bash
cd intellijob-backend

# Create and activate virtualenv
python -m venv venv
source venv/Scripts/activate            # Git Bash
# venv\Scripts\activate.bat            # cmd
# venv\Scripts\Activate.ps1            # PowerShell

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm

# Apply migrations
python manage.py migrate

# Seed job data from Adzuna (one-time or periodic)
python manage.py seed_jobs_adzuna --pages 20 --what "software engineer" --where "UK"

# Build market clusters (HDBSCAN + LLM naming) - runs automatically on first discovery request
# Or force rebuild:
python -c "from api.services.market_clustering import build_market_clusters; build_market_clusters(force_rebuild=True)"

# Run the dev server
python manage.py runserver
```

### 3. Frontend

```bash
cd intellijob-frontend
npm install
npm run dev          # http://localhost:5173
```

### 4. Ollama (Local LLM)

```bash
# Install: https://ollama.com/download
ollama serve
ollama pull gpt-oss:120b   # Used for roadmap, trajectory, phase plans, cluster naming
```

### 5. Environment Configuration

Create `intellijob-backend/.env`:

```env
# Adzuna API credentials (https://developer.adzuna.com/)
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key

# Django
DJANGO_SECRET_KEY=dev-only-not-secret-change-in-prod
DJANGO_DEBUG=True

# Ollama Cloud API (create key at https://ollama.com/settings/keys)
OLLAMA_API_KEY=your_ollama_cloud_key
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_BASE_URL=https://ollama.com/v1

# Cluster naming (uses same Ollama Cloud)
USE_LLM_CLUSTER_NAMING=true
LLM_NAMING_MODEL=gpt-oss:120b
LLM_NAMING_MAX_TOKENS=4000

# Tavily AI Search API (for learning resources)
TAVILY_API_KEY=your_tavily_key

# Market-driven pathways (true = HDBSCAN, false = fixed taxonomy)
USE_MARKET_PATHWAYS=true
```

---

## 🧪 Development

### Linting

```bash
# Backend — Ruff
cd intellijob-backend
ruff check .
ruff format .

# Frontend — ESLint
cd intellijob-frontend
npm run lint
```

### Tests

```bash
# Backend
cd intellijob-backend
python manage.py test

# Frontend
cd intellijob-frontend
npm run test
```

### Type Checking

```bash
cd intellijob-frontend
npx tsc --noEmit
```

---

## 🛠️ Tech Stack

| Layer                | Technology                                                                       |
| -------------------- | -------------------------------------------------------------------------------- |
| Frontend             | React 19, TypeScript, Vite, Tailwind CSS, ShadCN UI, Axios                       |
| Backend              | Django 5.2, Django REST Framework                                                |
| PDF Parsing          | PyMuPDF (`fitz`)                                                                 |
| NLP / NER            | spaCy 3.8 + custom `EntityRuler` (gazetteer)                                    |
| Embeddings           | SentenceTransformers `all-MiniLM-L6-v2` (384-dim)                               |
| Vector Search        | In-memory NumPy cosine similarity (file-based `.npz` + pickle cache); PostgreSQL/pgvector/HNSW planned for production |
| Clustering           | HDBSCAN + UMAP (market-driven pathway discovery)                                 |
| LLM (Roadmap/Names)  | Ollama Cloud OpenAI-compatible API (`gpt-oss:120b`)                             |
| Learning Resources   | Tavily Search API (cached 30 days)                                               |
| Linting              | Ruff (Python), ESLint (TypeScript)                                               |
| CI / CD              | GitLab CI                                                                        |

---

## 🔐 Privacy & Ethics

All résumé parsing, skill extraction, embedding, vector search, clustering, and LLM inference happen **locally** or via **Ollama Cloud** (no third-party candidate data storage). No résumé text, embeddings, or prompts are transmitted to uncontrolled third-party APIs. This is a deliberate design choice made to satisfy **GDPR data-minimisation** requirements and to avoid vendor lock-in for the dissertation's evaluation.

---

## 📚 Dissertation Context

- **Programme:** MSc IT+ (Information Technology), University of Glasgow
- **Repository:** `stgit.dcs.gla.ac.uk/msc-project-for-information-technology/2025/it-project-3171501k/intellijob`
- **Project type:** Final dissertation
- **Specification:** see [`intellijob-frontend/Specifications.txt`](intellijob-frontend/Specifications.txt)

---

## 📖 Design Decisions & Rationale

### 1. Market-Driven Clustering vs. Fixed Taxonomy

**Decision:** Use HDBSCAN on job embeddings to discover career pathways from actual market data, rather than a hand-curated fixed taxonomy.

**Why:**
- **Adaptability** — Job market evolves; clusters emerge from data, not human bias.
- **Granularity** — Discovers sub-specialisations (e.g., "AI Data Engineering" vs "Data Engineering (Lead)") that fixed taxonomies miss.
- **Explainability** — Each cluster has real job titles, keywords, and size as evidence.

**Alternatives Considered:**
| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Fixed taxonomy (14 pathways) | Predictable, zero infra, fast | Rigid, misses emerging roles, UK-market bias | Kept as **test fallback** (`USE_MARKET_PATHWAYS=false`) |
| K-Means + manual naming | Simple | Assumes spherical clusters, K unknown, poor on embeddings | Rejected |
| **HDBSCAN + UMAP + LLM naming** | Density-based, auto-K, handles noise, data-driven names | Heavier offline compute, needs LLM for naming | **Chosen** |

**Trade-offs Accepted:**
- Offline clustering takes ~60-90s on 1056 jobs (one-time, cached).
- LLM naming adds ~5-10s per rebuild (mitigated by 4000 token limit, system prompt).
- Cluster boundaries are probabilistic (HDBSCAN noise points excluded).

### 2. Dual-Mode Architecture (Targeted vs Discovery)

**Decision:** Two distinct analysis modes sharing the same embedding + skill extraction pipeline.

**Why:**
- **Targeted** serves users with a clear goal (classic skill-gap analysis).
- **Discovery** serves explorers/career changers (broader market view).
- Shared pipeline = DRY, consistent skill extraction, same match quality.

**Implementation:**
- `AnalyzeView.post()` branches on `target_title` presence.
- Discovery uses `_discovery_response_market()` with cluster ranking + lazy roadmaps.
- Targeted uses matcher directly + single roadmap.

### 3. Lazy-Loaded Roadmaps in Discovery Mode

**Decision:** Generate roadmap only for pathway #1 initially; others generated on-demand via `POST /api/pathway-roadmap/`.

**Why:**
- Prevents 3-4× LLM latency on initial request (30-120s vs 10-30s).
- Avoids queueing multiple LLM requests if user clicks rapidly.
- Frontend disables other pathway cards during generation.

**Alternatives Considered:**
| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Generate all upfront | Immediate availability | 3-4× latency, request pile-up | Rejected |
| **Lazy-load on click** | Fast initial response, controlled load | Extra API call per pathway | **Chosen** |
| Batch async with max_workers=2 | Parallel, some control | Still high initial latency | Rejected |

### 4. LLM Cluster Naming (Ollama Cloud)

**Decision:** Use the same Ollama Cloud model (`gpt-oss:120b`) for canonical cluster names via structured prompting.

**Why:**
- Consistent with roadmap generation (same infra, same model).
- Produces professional 2-4 word names ("Machine Learning Engineering") vs rule-based fallbacks ("Data Science / AI (Group 3)").
- System prompt enforces JSON-only output, preventing reasoning-token waste.

**Fallback Chain:**
1. LLM naming (Ollama Cloud, 4000 tokens, system prompt → JSON only)
2. Rule-based naming with deduplication (titles + keywords → differentiators)
3. "Group N" numeric suffix (last resort)

### 5. Career Trajectory (3-Phase Progression)

**Decision:** Synthesise a strategic 3-phase trajectory from skill gaps + matched jobs, not just a flat roadmap.

**Why:**
- Dissertation requirement: "strategic career advisor, not just task list".
- Phases map to real career progression: Entry → Mid → Senior.
- Each phase has objectives, titles, salary, required skills, unlock criteria.

**PhaseDetailModal** provides week-by-week plans via `POST /api/phase-plan/` with "Retry with AI" on fallback.

### 6. Learning Resources via Tavily

**Decision:** Live search for official docs/tutorials/courses, cached 30 days.

**Why:**
- Links rot; static lists become stale.
- Tavily returns high-quality, categorised results (documentation/tutorial/interactive).
- 30-day cache balances freshness with API costs.

---

## 🔮 Future Extensions (Roadmap)

### Phase 1: Personalisation & Persistence (Database Integration)
- **User Accounts & Profiles** — JWT auth, profile with uploaded CVs, extracted skills history.
- **Saved Roadmaps** — Persist generated roadmaps/trajectories per user; versioning for CV updates.
- **Progress Tracking** — Mark learning steps complete; visual progress on trajectory.
- **CV Versioning** — Compare skill extraction across CV uploads; show gap closure.

### Phase 2: Conversational Career Coach (Chatbot)
- **Context-Aware Chat** — "Given my roadmap, what should I focus on this week?"
- **RAG over User Context** — Inject user's skills, target pathway, current phase into chat prompt.
- **Proactive Nudges** — "You've completed 3/5 steps in Phase 1; ready for Phase 2 planning?"
- **Model** — Same Ollama Cloud, system prompt as career coach.

### Phase 3: Market Intelligence & Alerts
- **Trend Detection** — Re-cluster monthly; detect emerging clusters (new roles, declining demand).
- **Skill Demand Heatmap** — Visualise top skills by pathway over time.
- **Job Alert Integration** — Notify when new Adzuna postings match user's pathway.
- **Salary Benchmarking** — Track salary range shifts per pathway/phase.

### Phase 4: Collaborative & Social Features
- **Mentor Matching** — Connect users in Phase 2 with Phase 3 volunteers (opt-in).
- **Peer Cohorts** — Group users targeting same pathway; shared resources, accountability.
- **Portfolio Showcase** — Public project links per completed learning step.

### Phase 5: Enterprise / Institutional
- **University Dashboard** — Aggregate cohort analytics (skill gaps by programme, placement outcomes).
- **Employer Portal** — Post roles directly; get candidate skill-match scores.
- **White-Label API** — Embed IntelliJob analysis in partner career services.

---

## 🧪 Testing Strategy

| Layer | Approach | Coverage |
|-------|----------|----------|
| Skill Extraction | Unit tests with fixture PDFs | Gazetteer patterns, edge cases |
| Embedding + Search | Integration tests with seeded jobs | HNSW accuracy, latency |
| Clustering | Deterministic fixture (20 jobs) | Cluster stability, naming fallback |
| Roadmap Generation | Mock LLM client | JSON parsing, fallback, phase plans |
| API Views | DRF test client | Targeted/Discovery modes, lazy roadmap |
| Frontend | Vitest + React Testing Library | Component rendering, loading states, click disabling |

Run all: `python manage.py test` (backend) + `npm run test` (frontend).

---

## 📄 License

TBD — see dissertation submission guidelines.

---

## 👤 Author

MSc IT+ candidate, School of Computing Science, University of Glasgow.

---

## 📝 Changelog (Significant Changes)

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Aug 2026 | Market-driven clustering (HDBSCAN), LLM naming, Discovery mode, lazy roadmaps, 3-phase trajectory, Tavily resources, PhaseDetailModal |
| 1.0 | Jul 2026 | Fixed taxonomy (14 pathways), targeted mode only, single roadmap, basic skill-gap analysis |

---

*This document reflects the system as implemented for the MSc IT+ dissertation submission. Architecture decisions are justified in the dissertation thesis (Chapter 3: System Design).*