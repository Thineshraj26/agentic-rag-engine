from app.core.graph import build_graph

agent = build_graph()


def test_query(prompt: str):
    print(f"\n==========================================")
    print(f"Testing Query: '{prompt}'")
    print(f"==========================================")

    initial_state = {
        "query": prompt,
        "original_query": prompt,
        "documents": [],
        "generation": None,
        "loop_count": 0,
        "is_grounded": True,
        "needs_retrieval": False,
    }

    result = agent.invoke(initial_state)
    print(f"\n[Final Response]:\n{result['generation']}")
    print(f"\n[Docs Used]: {len(result['documents'])} chunks")
    print(f"[Loop Rewrites]: {result['loop_count']}")


if __name__ == "__main__":
    # Test 1: Should route to direct_answer without vector retrieval
    test_query("Hello! How's your day going?")

    # Test 2: In-scope technical query targeting MCP chunk
    test_query("What is the Model Context Protocol and what does it enable?")

    # Test 3: Technical query targeting LangGraph state chunk
    test_query("How is state passed across nodes in LangGraph?")