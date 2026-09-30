import os
import pytest
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.graph import build_graph

load_dotenv()

# Judge LLM
judge_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=150,  # Structured eval decisions only require ~50 tokens
)

# Shared graph instance
agent = build_graph()


# ==========================================
# Evaluation Metric Schemas
# ==========================================

class FaithfulnessScore(BaseModel):
    """Measures if the response is completely faithful to the retrieved documents."""
    is_faithful: bool = Field(
        description="True if EVERY factual statement is backed by context, False if hallucinated.")
    reason: str = Field(description="Brief explanation of the judgment.")


class RelevanceScore(BaseModel):
    """Measures if the response answers the original question asked."""
    is_relevant: bool = Field(description="True if the response directly addresses the question, False otherwise.")
    reason: str = Field(description="Brief explanation of the judgment.")


# ==========================================
# Evaluator Functions
# ==========================================

def evaluate_faithfulness(question: str, context: str, answer: str) -> FaithfulnessScore:
    structured_judge = judge_llm.with_structured_output(FaithfulnessScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an impartial judge evaluating whether an AI assistant's answer is faithful to the context.\n"
            "An answer is FAITHFUL if all facts in the answer can be directly derived from the context.\n"
            "An answer is NOT FAITHFUL if it invents external facts not found in the context."
        )),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}")
    ])
    chain = prompt | structured_judge
    return chain.invoke({"context": context, "question": question, "answer": answer})


def evaluate_relevance(question: str, answer: str) -> RelevanceScore:
    structured_judge = judge_llm.with_structured_output(RelevanceScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an impartial judge evaluating whether an answer directly addresses the question.\n"
            "An answer is RELEVANT if it directly responds to what was asked."
        )),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}")
    ])
    chain = prompt | structured_judge
    return chain.invoke({"question": question, "answer": answer})


# ==========================================
# Golden Dataset Test Cases
# ==========================================

GOLDEN_TEST_CASES = [
    {
        "query": "What is the Model Context Protocol and what does it enable?",
        "expected_topic": "MCP"
    },
    {
        "query": "How is state passed across nodes in LangGraph?",
        "expected_topic": "LangGraph"
    }
]


@pytest.mark.parametrize("test_case", GOLDEN_TEST_CASES)
def test_agent_faithfulness_and_relevance(test_case):
    time.sleep(2) # Prevent back-to-back burst rate limits
    query = test_case["query"]

    # 1. Run the agent workflow
    state_input = {
        "query": query,
        "original_query": query,
        "documents": [],
        "generation": None,
        "loop_count": 0,
        "is_grounded": True,
        "needs_retrieval": False,
    }
    output = agent.invoke(state_input)

    answer = output["generation"]
    docs = output["documents"]

    assert len(docs) > 0, f"Expected retrieved documents for query: '{query}'"

    context_str = "\n".join([d.content for d in docs])

    # 2. Score Faithfulness (No Hallucination)
    faith_result = evaluate_faithfulness(question=query, context=context_str, answer=answer)
    print(f"\n[Eval] Faithfulness: {faith_result.is_faithful} | Reason: {faith_result.reason}")
    assert faith_result.is_faithful, f"Faithfulness check failed: {faith_result.reason}"

    # 3. Score Relevance
    rel_result = evaluate_relevance(question=query, answer=answer)
    print(f"[Eval] Relevance: {rel_result.is_relevant} | Reason: {rel_result.reason}")
    assert rel_result.is_relevant, f"Relevance check failed: {rel_result.reason}"