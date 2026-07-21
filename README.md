# IntelliJob

> A decoupled, offline-first career analytics platform that turns a candidate's PDF résumé into a personalised, AI-generated skill-gap roadmap — built for the MSc IT+ dissertation at the **University of Glasgow**.

IntelliJob shifts away from generic live job-board scrapers. It acts as a **deterministic, strategic career advisor**: it parses résumés, extracts skills, performs sub-50 ms semantic vector search against a frozen UK software-engineering job dataset, and synthesises a context-grounded learning plan via a **local** Retrieval-Augmented Generation (RAG) pipeline. **No candidate data ever leaves the machine.**

---

## ✨ Features

- 📄 **Deterministic PDF résumé parsing** via `pymupdf4llm` (clean Markdown extraction, layout-aware).
- 🧠 **Custom NLP skill tokenisation** using `spaCy` + a hand-tuned `EntityRuler` for technical entities (`SKILL`).
- 🔢 **384-dim semantic embeddings** with `SentenceTransformers` (`all-MiniLM-L6-v2`).
- ⚡ **Sub-50 ms vector similarity search** against PostgreSQL using the `pgvector` extension and a pre-computed **HNSW** graph index.
- 🗺️ **Local RAG roadmapping** with `LangChain` orchestrating a fully local `Ollama` model (Llama 3 / Mistral) — produces a Markdown skill-gap plan citing the top matched job specifications.
- 🔒 **100 % local processing** — no OpenAI, no cloud APIs, GDPR-friendly by design.
- 🧱 **Decoupled architecture** — React SPA + Django REST Framework API + PostgreSQL.

---

## 🏗️ Architecture

A three-tier system:

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1 — React SPA (Vite + TS + Tailwind + ShadCN)            │
│            Axios → Django endpoints                             │
└─────────────────────────────────────────────────────────────────┘
                              │  HTTP / JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2 — Django REST Framework (Python 3.11+)                 │
│   PyMuPDF ─▶ spaCy NER ─▶ SentenceTransformers ─▶ LangChain    │
│                                       (orchestration)           │
└─────────────────────────────────────────────────────────────────┘
                              │  pgvector SQL
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 3 — PostgreSQL + pgvector + HNSW index                   │
│            Pre-vectorised static UK job-spec dataset            │
└─────────────────────────────────────────────────────────────────┘
```

### End-to-End Pipeline

1. **Offline preparation (one-time)**
   - A static, uncurated research dataset of UK software-engineering job ads (e.g. a Kaggle corpus) is seeded into PostgreSQL.
   - A background script embeds every job spec with `all-MiniLM-L6-v2`.
   - An HNSW graph index is built over the embedding column.

2. **Real-time user loop (per request)**
   - User uploads a PDF CV and a target job title in the React dashboard.
   - Django extracts clean text with PyMuPDF.
   - spaCy extracts skill tokens (Django, React, Docker, PostgreSQL, …).
   - The user's skill vector queries PostgreSQL via `pgvector` cosine distance — top 3–5 specs returned in < 50 ms.
   - LangChain injects the candidate's skills + the matched specs into a bounded prompt.
   - Ollama (local) generates a Markdown roadmap highlighting missing competencies.

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
│   └── api/                # Application logic: models, views, endpoints
└── intellijob-frontend/    # React + TypeScript SPA (Vite)
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    └── src/
```

---

## 🚀 Getting Started

### Prerequisites

| Tool                | Version           | Notes                                               |
| ------------------- | ----------------- | --------------------------------------------------- |
| Python              | 3.11+             | Backend runtime                                     |
| Node.js             | 20 LTS or newer   | Frontend runtime                                    |
| PostgreSQL          | 15+               | With the `pgvector` extension installed             |
| Ollama              | latest            | Local LLM runtime (`ollama pull llama3` etc.)       |

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

# (One-time) seed the job dataset and build the HNSW index
python manage.py seed_jobs --source data/jobs.csv
python manage.py build_hnsw_index

# Run the dev server
python manage.py runserver
```

### 3. Frontend

```bash
cd intellijob-frontend
npm install
npm run dev          # http://localhost:5173
```

### 4. Local LLM (Ollama)

```bash
# Install: https://ollama.com/download
ollama serve
ollama pull llama3
```

---

## 🧪 Development

### Linting (per project spec, Section 5)

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
```

---

## 🛠️ Tech Stack

| Layer        | Technology                                                              |
| ------------ | ----------------------------------------------------------------------- |
| Frontend     | React 19, TypeScript, Vite, Tailwind CSS, ShadCN UI, Axios               |
| Backend      | Django 5.2, Django REST Framework                                       |
| PDF parsing  | PyMuPDF (`pymupdf4llm`)                                                 |
| NLP / NER    | spaCy 3.8 + custom `EntityRuler`                                        |
| Embeddings   | SentenceTransformers `all-MiniLM-L6-v2` (384-dim)                       |
| Vector DB    | PostgreSQL 15+, `pgvector` extension, HNSW index                        |
| RAG / LLM    | LangChain + LangGraph, local Ollama (Llama 3 / Mistral)                 |
| Linting      | Ruff (Python), ESLint (TypeScript)                                      |
| CI / CD      | GitLab CI                                                               |

---

## 🔐 Privacy & Ethics

All résumé parsing, skill extraction, embedding, vector search, and LLM inference happen **locally** on the user's machine. No candidate data, embeddings, or prompts are transmitted to any third-party API. This is a deliberate design choice made to satisfy **GDPR data-minimisation** requirements and to avoid vendor lock-in for the dissertation's evaluation.

---

## 📚 Dissertation Context

- **Programme:** MSc IT+ (Information Technology), University of Glasgow
- **Repository:** `stgit.dcs.gla.ac.uk/msc-project-for-information-technology/2025/it-project-3171501k/intellijob`
- **Project type:** Final dissertation
- **Specification:** see [`intellijob-frontend/Specifications.txt`](intellijob-frontend/Specifications.txt)

---

## 📄 License

TBD — see dissertation submission guidelines.

---

## 👤 Author

MSc IT+ candidate, School of Computing Science, University of Glasgow.
