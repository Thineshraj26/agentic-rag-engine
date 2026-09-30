@"
# Enterprise Agentic RAG Engine

A production-grade, stateful Retrieval-Augmented Generation (RAG) platform featuring self-correcting agent loops, asynchronous background document processing, hybrid vector storage, automated LLM-as-a-Judge evaluations, and a dark-mode enterprise console.

---

## Architecture Overview

\`\`\`
                      +-----------------------------------------+
                      |         Streamlit Console (:8501)       |
                      +--------------------+--------------------+
                                           |
                                           | HTTP Requests
                                           v
                      +--------------------+--------------------+
                      |       FastAPI Backend API (:8000)       |
                      +---------+---------------------+---------+
                                |                     |
                  [Query Path]  |                     |  [Ingestion Path]
                                v                     v
          +---------------------+----------+    +-----+--------------------+
          |    LangGraph Agentic Engine    |    |   Redis Broker (:6379)   |
          | - Intent Router (Direct vs RAG)|    +--------------+-----------+
          | - Document Retrieval & Grading |                   |
          | - Dynamic Query Rewriting Loop |                   v
          | - Hallucination Grounding Eval |    +--------------+-----------+
          +---------------------+----------+    |   Celery Background      |
                                |               |   Ingestion Worker       |
                                |               | - PDF Extraction (pypdf) |
                                |               | - Semantic Chunking      |
                                |               | - FastEmbed Vectorizer   |
                                +-------+-------+--------------+-----------+
                                        |                      |
                                        v                      v
                      +-----------------+----------------------+
                      |      PostgreSQL 16 + pgvector (:5432)  |
                      | - Parent Document Relational Tracking  |
                      | - 384-dim HNSW Vector Cosine Index     |
                      | - Full-Text Search Index (tsvector GIN)|
                      +----------------------------------------+
\`\`\`

---

## Key Features

- **Stateful Agentic Graph (LangGraph):** Employs a multi-node cyclical state machine with intentional routing, retrieval grading, dynamic query rewriting loops when retrieved context is inadequate, and hallucination verification.
- **Asynchronous Document Processing:** Decoupled PDF ingestion via Celery and Redis to prevent blocking HTTP threads during large file uploads.
- **FastEmbed Local Embeddings:** Generates 384-dimensional dense vectors using \`BAAI/bge-small-en-v1.5\` locally, removing external API latency, embedding rate limits, and operational costs.
- **PostgreSQL + pgvector:** Hybrid database architecture combining relational parent-document state tracking, HNSW vector cosine distance indexing, and full-text keyword indexing.
- **Strict Free-Tier Rate Management:** Optimized output token budgeting (\`max_tokens\`) configured for Groq free-tier limits (1,000 OTPM).
- **Automated CI/CD Evaluation Harness:** Built-in LLM-as-a-Judge test suite assessing **Faithfulness** (hallucination detection) and **Answer Relevance** using \`pytest\`.
- **Spotify-Inspired Enterprise Interface:** Dark-mode Streamlit dashboard with real-time ingestion status polling, collapsible cited evidence cards, cosine similarity metrics, and routing badges.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph, LangChain Core |
| **API Framework** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Database** | PostgreSQL 16, pgvector |
| **Embeddings** | FastEmbed (\`BAAI/bge-small-en-v1.5\`) |
| **LLM Inference** | Groq Cloud (\`qwen/qwen3.8-27b\`) |
| **Task Queue** | Celery, Redis |
| **Document Parsing** | pypdf |
| **Testing & Evals** | pytest, Pydantic |
| **Containerization** | Docker, Docker Compose |

---

## Project Structure

\`\`\`text
agentic-rag-engine/
├── app/
│   ├── core/
│   │   └── graph.py             # LangGraph state machine, nodes, and routing logic
│   ├── evals/
│   │   └── test_rag_eval.py     # Automated LLM-as-a-judge test suite
│   ├── retrieval/
│   │   ├── store.py             # pgvector schema, migrations, and vector search
│   │   └── tasks.py             # Celery async ingestion and PDF parsing worker
│   └── main.py                  # FastAPI REST endpoints and status polling
├── docker-compose.yml           # Multi-container orchestration (Postgres + Redis)
├── frontend.py                  # Spotify-themed Streamlit web console
├── run_agent.py                 # CLI agent interaction and verification runner
├── seed_db.py                   # Schema initialization and test chunk seeder
├── requirements.txt             # Environment dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Ignored files, virtual environments, and caches
└── README.md                    # System documentation
\`\`\`

---

## Getting Started

### 1. Prerequisites

- Docker Desktop installed and running.
- Python 3.11 or later installed.
- A free API key from [Groq Console](https://console.groq.com/).

### 2. Clone the Repository & Setup Environment

\`\`\`bash
git clone https://github.com/<your-username>/agentic-rag-engine.git
cd agentic-rag-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (Command Prompt / PowerShell):
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 3. Configure Secrets

Copy the example environment template:

\`\`\`bash
cp .env.example .env
\`\`\`

Open \`.env\` and insert your credentials:

\`\`\`ini
# PostgreSQL (pgvector)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db

# Redis Message Broker
REDIS_URL=redis://localhost:6379/0

# Groq Cloud API
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
\`\`\`

### 4. Start Infrastructure Containers

Spin up the pgvector-enabled PostgreSQL database and Redis message broker:

\`\`\`bash
docker compose up -d
\`\`\`

Verify both containers are running:

\`\`\`bash
docker ps
\`\`\`

### 5. Initialize Schema & Seed Verification Data

Run the database setup script to apply schemas, configure HNSW indexes, and populate initial test knowledge chunks:

\`\`\`bash
python seed_db.py
\`\`\`

---

## Running the Automated Evaluation Suite

The application includes an LLM-as-a-judge evaluation harness that checks every response against two core metrics:
1. **Faithfulness:** Verifies that no statement in the response hallucinates beyond the retrieved context.
2. **Relevance:** Confirms the synthesized answer directly addresses the original prompt.

Run the test suite:

\`\`\`bash
pytest -s app/evals/test_rag_eval.py
\`\`\`

---

## Running the Application

To run the complete system locally, launch each service in its own terminal window with the \`.venv\` activated:

### Terminal 1: Celery Background Worker
Handles non-blocking PDF text extraction, chunking, and local embedding generation.

\`\`\`bash
# Windows:
celery -A app.retrieval.tasks.celery_app worker --loglevel=info --pool=solo

# macOS / Linux:
# celery -A app.retrieval.tasks.celery_app worker --loglevel=info
\`\`\`

### Terminal 2: FastAPI Backend Server
Serves the core API, document upload pipeline, and agent invocation routes.

\`\`\`bash
uvicorn app.main:app --reload --port 8000
\`\`\`
- API Base: \`http://localhost:8000\`
- Interactive Swagger Documentation: \`http://localhost:8000/docs\`

### Terminal 3: Streamlit Interface
Launches the dark-mode knowledge console for document uploads and chat queries.

\`\`\`bash
streamlit run frontend.py
\`\`\`
- Web Console: \`http://localhost:8501\`

---

## API Reference

### Health Check
\`\`\`http
GET /health
\`\`\`
Returns backend operational status (\`{"status": "healthy"}\`).

### Upload Document
\`\`\`http
POST /api/documents/upload
Content-Type: multipart/form-data
\`\`\`
Uploads a PDF file. Dispatches an async ingestion job to the Celery queue and returns a \`document_id\`.

### Poll Ingestion Status
\`\`\`http
GET /api/documents/{document_id}/status
\`\`\`
Returns processing stage (\`pending\` -> \`processing\` -> \`completed\` / \`failed\`) along with total indexed chunk counts.

### Execute Agent Query
\`\`\`http
POST /api/query
Content-Type: application/json

{
  "query": "What is the Model Context Protocol?"
}
\`\`\`
Invokes the LangGraph decision workflow and returns the grounded answer, rewrite iterations, routing details, and cited source chunks with cosine similarity scores.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
"@ | Set-Content -Path README.md -Encoding utf8