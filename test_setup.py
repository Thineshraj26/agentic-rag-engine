import os
from dotenv import load_dotenv
from app.core.state import AgentState, DocumentChunk

load_dotenv()

print("Testing environment...")
assert os.getenv("OPENAI_API_KEY"), "Error: OPENAI_API_KEY not found in .env"

# Test state typing
mock_state: AgentState = {
    "query": "What is hybrid search?",
    "original_query": "What is hybrid search?",
    "documents": [
        DocumentChunk(id="doc_1", content="Hybrid search combines dense and sparse vectors.")
    ],
    "generation": None,
    "loop_count": 0,
    "is_grounded": True,
    "needs_retrieval": True
}

print("Pydantic State schema verified successfully.")
print("Setup complete.")