# Enterprise Agentic RAG Engine

A production-grade, stateful Retrieval-Augmented Generation (RAG) platform featuring self-correcting agent loops, asynchronous background document processing, hybrid vector storage, automated LLM-as-a-Judge evaluations, and a dark-mode enterprise console.

---

## Architecture Overview

```mermaid
flowchart TD
    UI["Streamlit Console (:8501)"]
    API["FastAPI Backend API (:8000)"]

    subgraph QueryPath ["Query Flow"]
        Agent["LangGraph Agentic Engine<br/>• Intent Router<br/>• Retrieval & Grading<br/>• Query Rewrite Loop<br/>• Grounding Evaluation"]
    end

    subgraph IngestionPath ["Async Ingestion Flow"]
        Redis["Redis Broker (:6379)"]
        Worker["Celery Ingestion Worker<br/>• PDF Extraction (pypdf)<br/>• Semantic Chunking<br/>• FastEmbed Embeddings"]
    end

    DB[("PostgreSQL 16 + pgvector (:5432)<br/>• Parent Document Tracking<br/>• 384-dim HNSW Vector Index<br/>• Full-Text Search (tsvector GIN)")]

    UI -->|HTTP Requests| API
    API -->|Synchronous Query| Agent
    API -->|Async Job Dispatch| Redis
    Redis --> Worker
    Agent <-->|Vector Retrieval| DB
    Worker -->|Batch Inserts| DB

    classDef ui fill:#181818,stroke:#1DB954,stroke-width:2px,color:#FFFFFF;
    classDef api fill:#242424,stroke:#555,stroke-width:1px,color:#FFFFFF;
    classDef worker fill:#1e1e1e,stroke:#444,stroke-width:1px,color:#CCCCCC;
    classDef db fill:#121212,stroke:#1DB954,stroke-width:2px,color:#FFFFFF;

    class UI ui;
    class API api;
    class Agent,Redis,Worker worker;
    class DB db;
```

---

## Key Features

- **Stateful Agentic Graph (LangGraph):** Employs a multi-node cyclical state machine with intentional routing, retrieval grading, dynamic query rewriting loops when retrieved context is inadequate, and hallucination verification.
- **Asynchronous Document Processing:** Decoupled PDF ingestion via Celery and Redis to prevent blocking HTTP threads during large file uploads.
- **FastEmbed Local Embeddings:** Generates 384-dimensional dense vectors using `BAAI/bge-small-en-v1.5` locally, removing external API latency, embedding rate limits, and operational costs.
- **PostgreSQL + pgvector:** Hybrid database architecture combining relational parent-document state tracking, HNSW vector cosine distance indexing, and full-text keyword indexing.
- **Strict Free-Tier Rate Management:** Optimized output token budgeting (`max_tokens`) configured for Groq free-tier limits (1,000 OTPM).
- **Automated CI/CD Evaluation Harness:** Built-in LLM-as-a-Judge test suite assessing **Faithfulness** (hallucination detection) and **Answer Relevance** using `pytest`.
- **Spotify-Inspired Enterprise Interface:** Dark-mode Streamlit dashboard with real-time ingestion status polling, collapsible cited evidence cards, cosine similarity metrics, and routing badges.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph, LangChain Core |
| **API Framework** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Database** | PostgreSQL 16, pgvector |
| **Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5`) |
| **LLM Inference** | Groq Cloud (`qwen/qwen3.8-27b`) |
| **Task Queue** | Celery, Redis |
| **Document Parsing** | pypdf |
| **Testing & Evals** | pytest, Pydantic |
| **Containerization** | Docker, Docker Compose |

---

## Project Structure

```text
agentic-rag-engine/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI REST endpoints and background job trigger
│   ├── core/
│   │   ├── __init__.py
│   │   └── graph.py             # LangGraph state machine, nodes, and routing logic
│   ├── evals/
│   │   ├── __init__.py
│   │   └── test_rag_eval.py     # Automated LLM-as-a-judge test harness (Faithfulness/Relevance)
│   └── retrieval/
│       ├── __init__.py
│       ├── store.py             # pgvector schema, migrations, and dense/hybrid search
│       └── tasks.py             # Celery application and async PDF parsing task
├── docker-compose.yml           # Local infrastructure stack (Postgres + pgvector & Redis)
├── frontend.py                  # Spotify-themed Streamlit dark console
├── run_agent.py                 # CLI test runner for direct agent verification
├── seed_db.py                   # Database schema initialization and golden dataset seeder
├── requirements.txt             # Frozen production dependencies
├── .env.example                 # Environment variable configuration template
├── .gitignore                   # Exclusions for virtual environments, secrets, and caches
└── README.md                    # Project documentation
```

---

## Getting Started

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- Python 3.11 or later.
- A free API key from [Groq Console](https://console.groq.com/).

### 2. Clone the Repository & Setup Environment

```bash
git clone [https://github.com/](https://github.com/)<your-username>/agentic-rag-engine.git
cd agentic-rag-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell / Command Prompt):
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets

Copy the template environment file:

```bash
cp .env.example .env
```

Open `.env` and set your credentials:

```ini
# PostgreSQL (pgvector)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db

# Redis Message Broker
REDIS_URL=redis://localhost:6379/0

# Groq Cloud API
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 4. Start Infrastructure Containers

Launch the PostgreSQL container (with `pgvector` enabled) and Redis broker:

```bash
docker compose up -d
```

Verify that both containers are up:

```bash
docker ps
```

### 5. Initialize Schema & Seed Golden Chunks

Run the migration and seeding script to initialize the relational tables, build HNSW vector indexes, and insert golden evaluation data:

```bash
python seed_db.py
```

---

## Running the Automated Evaluation Suite

The platform includes an automated LLM-as-a-Judge evaluation test suite built with `pytest` that benchmarks the pipeline against golden context:
1. **Faithfulness:** Asserts that every statement in the generated answer is directly backed by retrieved context (hallucination guard).
2. **Relevance:** Asserts that the response directly answers the user's specific prompt.

Run the test suite:

```bash
pytest -s app/evals/test_rag_eval.py
```

---

## Running the Application

To run the complete system, launch each service in its own terminal window with the virtual environment activated (`.venv\Scripts\activate`):

### Terminal 1: Celery Background Worker
Processes uploaded PDFs asynchronously, chunks content, computes local vector embeddings, and bulk inserts into PostgreSQL.

```bash
# Windows:
celery -A app.retrieval.tasks.celery_app worker --loglevel=info --pool=solo

# macOS / Linux:
# celery -A app.retrieval.tasks.celery_app worker --loglevel=info
```

### Terminal 2: FastAPI Backend Server
Hosts the API endpoints, coordinates asynchronous job handoffs to Redis, and executes the LangGraph workflow.

```bash
uvicorn app.main:app --reload --port 8000
```
- API Base: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### Terminal 3: Streamlit Interface
Launches the dark-mode knowledge console for document uploads and chat queries.

```bash
streamlit run frontend.py
```
- Web Console: `http://localhost:8501`

---

## API Reference

### Health Check
```http
GET /health
```
Returns backend operational status (`{"status": "healthy"}`).

### Upload Document
```http
POST /api/documents/upload
Content-Type: multipart/form-data
```
Uploads a PDF file. Dispatches an async ingestion job to the Celery queue and returns a `document_id`.

### Poll Ingestion Status
```http
GET /api/documents/{document_id}/status
```
Returns processing stage (`pending` -> `processing` -> `completed` / `failed`) along with total indexed chunk counts.

### Execute Agent Query
```http
POST /api/query
Content-Type: application/json

{
  "query": "What is the Model Context Protocol?"
}
```
Invokes the LangGraph decision workflow and returns the grounded answer, rewrite iterations, routing details, and cited source chunks with cosine similarity scores.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
