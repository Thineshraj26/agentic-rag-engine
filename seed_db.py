import os
from dotenv import load_dotenv
from app.retrieval.store import (
    init_db,
    create_document_record,
    update_document_status,
    insert_document_chunks,
    hybrid_search
)

load_dotenv()

# 1. Initialize tables and vector indexes
init_db()

# 2. Register parent seed document
SEED_DOC_ID = "doc_seed_knowledge_base"
create_document_record(doc_id=SEED_DOC_ID, filename="seed_docs.txt")

# 3. Golden chunks required by the evaluation suite
sample_chunks = [
    {
        "id": "chunk_mcp_overview",
        "content": (
            "Model Context Protocol (MCP) is an open standard that enables AI models "
            "to securely connect to external tools, data sources, and client environments."
        ),
        "metadata": {"topic": "MCP", "source": "seed_docs.txt"}
    },
    {
        "id": "chunk_langgraph_state",
        "content": (
            "In LangGraph, state is passed across nodes as a typed dictionary or Pydantic model, "
            "allowing multi-agent loops, state checkpointing, and conditional routing."
        ),
        "metadata": {"topic": "LangGraph", "source": "seed_docs.txt"}
    },
    {
        "id": "chunk_hybrid_search",
        "content": (
            "Hybrid search combines dense vector retrieval with traditional sparse BM25 keyword matching "
            "to optimize both semantic understanding and exact keyword precision."
        ),
        "metadata": {"topic": "Search", "source": "seed_docs.txt"}
    }
]

# 4. Insert chunks linked to the parent document and mark status ready
insert_document_chunks(doc_id=SEED_DOC_ID, chunks=sample_chunks)
update_document_status(doc_id=SEED_DOC_ID, status="completed", chunk_count=len(sample_chunks))

print("Seeded database with sample chunks successfully.")

# 5. Sanity check search
results = hybrid_search("What is the Model Context Protocol?")
if results:
    print(f"Verification query found {len(results)} chunks. Top score: {results[0]['score']:.4f}")
else:
    print("Verification query returned no results.")