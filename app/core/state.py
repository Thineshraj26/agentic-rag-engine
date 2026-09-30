from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete context chunk retrieved from the vector store."""
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    relevance_score: Optional[float] = None


class AgentState(TypedDict):
    """
    Shared state across all nodes in the LangGraph workflow.
    Every node receives this state, executes its logic, and returns updates.
    """
    query: str                       # The active search query (may be rewritten by the agent)
    original_query: str              # The raw user input (kept intact for final generation)
    documents: List[DocumentChunk]   # Chunks that survived grading and relevance filtering
    generation: Optional[str]        # Final synthesized answer
    loop_count: int                  # Safety counter to prevent infinite query-rewriting loops
    is_grounded: bool                # Flag set by the hallucination detection node
    needs_retrieval: bool            # Flag set by the initial router node