import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.core.state import AgentState, DocumentChunk
from app.retrieval.store import hybrid_search

load_dotenv()

# High-speed free tier LPU inference with strict tool/JSON support
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=400,  # Keeps generation comfortably under Groq's 1000 OTPM free tier limit
)


# ==========================================
# 1. Structured Output Schemas
# ==========================================

class RouteQuery(BaseModel):
    """Route user query to database retrieval or direct response."""
    destination: Literal["vectorstore", "direct"] = Field(
        description="Choose 'vectorstore' for technical questions or definitions; 'direct' for greetings and chit-chat."
    )


class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    is_relevant: bool = Field(
        description="True if the document contains semantic meaning relevant to the query, False otherwise."
    )


# ==========================================
# 2. Graph Node Functions
# ==========================================

def router_node(state: AgentState) -> dict:
    """Classifies if query requires knowledge-base retrieval."""
    structured_router = llm.with_structured_output(RouteQuery)
    system_prompt = (
        "You are an expert at routing user questions. "
        "Use 'vectorstore' for technical questions, definitions, protocols, or architecture. "
        "Use 'direct' for greetings, casual chit-chat, or conversational remarks."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    chain = prompt | structured_router
    decision = chain.invoke({"query": state["query"]})

    return {
        "needs_retrieval": (decision.destination == "vectorstore")
    }


def direct_answer_node(state: AgentState) -> dict:
    """Answers conversational queries without database retrieval."""
    response = llm.invoke(f"Answer the user's conversational remark politely: {state['original_query']}")
    return {"generation": response.content}


def retrieve_node(state: AgentState) -> dict:
    """Queries pgvector using the current search query."""
    raw_results = hybrid_search(query=state["query"], top_k=3)
    chunks = [
        DocumentChunk(
            id=r["id"],
            content=r["content"],
            metadata=r["metadata"],
            relevance_score=r["score"]
        )
        for r in raw_results
    ]
    return {"documents": chunks}


def grade_documents_node(state: AgentState) -> dict:
    """Evaluates retrieved chunks and filters out irrelevant noise."""
    structured_grader = llm.with_structured_output(GradeDocument)
    system_prompt = (
        "You are an evaluator assessing relevance of a retrieved document to a user query. "
        "If the document contains keywords or semantic meaning related to the question, grade it as relevant."
    )
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User query: {query}\n\nDocument snippet: {document}")
    ])
    grader_chain = grade_prompt | structured_grader

    filtered_docs = []
    for doc in state["documents"]:
        score = grader_chain.invoke({"query": state["query"], "document": doc.content})
        if score.is_relevant:
            filtered_docs.append(doc)

    return {"documents": filtered_docs}


def rewrite_query_node(state: AgentState) -> dict:
    """Rewrites the query to improve vector retrieval on the next iteration."""
    system_prompt = (
        "You are an AI assistant that reformulates queries for a semantic search engine. "
        "Analyze the user's question and rewrite it to be clearer, more specific, and keyword-rich."
    )
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Initial question: {query}\nFormulate an improved search query:")
    ])
    chain = rewrite_prompt | llm
    response = chain.invoke({"query": state["query"]})

    return {
        "query": response.content.strip(),
        "loop_count": state["loop_count"] + 1
    }


def generate_node(state: AgentState) -> dict:
    """Synthesizes the final grounded answer using only validated documents."""
    context = "\n\n".join([f"[{i + 1}] {doc.content}" for i, doc in enumerate(state["documents"])])
    prompt = (
        f"You are a helpful AI assistant. Answer the question using ONLY the provided context snippets.\n"
        f"If the answer is not contained in the context, say 'I do not have sufficient information in the knowledge base.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['original_query']}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    return {"generation": response.content}


# ==========================================
# 3. Conditional Routing Edges
# ==========================================

def route_after_router(state: AgentState) -> Literal["retrieve", "direct_answer"]:
    if state["needs_retrieval"]:
        return "retrieve"
    return "direct_answer"


def route_after_grading(state: AgentState) -> Literal["generate", "rewrite_query"]:
    if len(state["documents"]) > 0:
        return "generate"

    if state["loop_count"] < 2:
        return "rewrite_query"

    return "generate"


# ==========================================
# 4. Graph Assembly
# ==========================================

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer"
        }
    )
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query"
        }
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()