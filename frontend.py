import time
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise Knowledge Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Spotify-inspired dark palette and clean typography
st.markdown("""
<style>
    /* Global surface overrides */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #282828;
    }

    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Primary Spotify green buttons */
    div.stButton > button:first-child {
        background-color: #1DB954;
        color: #000000;
        border: none;
        border-radius: 500px;
        padding: 0.55rem 1.6rem;
        font-weight: 700;
        font-size: 0.875rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        transition: transform 0.15s ease, background-color 0.15s ease;
    }

    div.stButton > button:first-child:hover {
        background-color: #1ED760;
        color: #000000;
        transform: scale(1.02);
    }

    /* Chat message containers */
    .stChatMessage {
        background-color: #181818;
        border: 1px solid #242424;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background-color: #282828;
        color: #1DB954;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #1DB954;
        color: #000000;
    }

    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-green {
        background-color: rgba(29, 185, 84, 0.15);
        color: #1DB954;
        border: 1px solid #1DB954;
    }
    .badge-gray {
        background-color: #242424;
        color: #B3B3B3;
        border: 1px solid #333333;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1A1A1A !important;
        border: 1px solid #282828 !important;
        border-radius: 6px !important;
        color: #B3B3B3 !important;
        font-weight: 600;
    }

    /* Input text box */
    .stChatInputContainer {
        border-color: #333333;
        background-color: #181818;
    }

    .stChatInputContainer:focus-within {
        border-color: #1DB954;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }

    /* File uploader dropzone */
    [data-testid="stFileUploader"] {
        background-color: #181818;
        border: 1px dashed #333333;
        border-radius: 8px;
        padding: 1rem;
    }

    hr {
        border-color: #282828;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Session State Initialization
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# ==========================================
# Sidebar: Document Management
# ==========================================
with st.sidebar:
    st.markdown("### Document Ingestion")
    st.caption("Upload technical reports, manuals, or PDF documents to index into the knowledge base.")

    uploaded_file = st.file_uploader("Select PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("Upload & Ingest", use_container_width=True):
            with st.spinner("Dispatching ingestion payload..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{API_BASE_URL}/api/documents/upload", files=files)

                    if res.status_code == 200:
                        data = res.json()
                        doc_id = data["document_id"]
                        filename = data["filename"]

                        status_placeholder = st.empty()
                        progress_bar = st.progress(10)

                        completed = False
                        attempts = 0
                        while not completed and attempts < 60:
                            time.sleep(1.5)
                            status_res = requests.get(f"{API_BASE_URL}/api/documents/{doc_id}/status")

                            if status_res.status_code == 200:
                                status_data = status_res.json()
                                current_status = status_data["status"]

                                if current_status == "processing":
                                    progress_bar.progress(50)
                                    status_placeholder.info("Processing segments and computing vectors...")
                                elif current_status == "completed":
                                    progress_bar.progress(100)
                                    chunk_count = status_data["chunk_count"]
                                    status_placeholder.success(f"Ingested {chunk_count} chunks successfully.")
                                    st.session_state.uploaded_docs.append({
                                        "id": doc_id,
                                        "filename": filename,
                                        "chunks": chunk_count
                                    })
                                    completed = True
                                elif current_status == "failed":
                                    status_placeholder.error(f"Ingestion failed: {status_data.get('error_message')}")
                                    break
                            attempts += 1
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection failure: {e}")

    st.markdown("---")
    st.markdown("### Active Knowledge Sources")
    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.markdown(
                f"<div style='padding: 8px 12px; background-color: #181818; border-radius: 6px; margin-bottom: 6px; border: 1px solid #282828;'>"
                f"<div style='font-size: 0.85rem; font-weight: 600; color: #FFFFFF;'>{doc['filename']}</div>"
                f"<div style='font-size: 0.75rem; color: #1DB954;'>{doc['chunks']} indexed chunks</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.caption("No custom documents uploaded in current session.")

# ==========================================
# Main Chat Surface
# ==========================================
st.markdown("<h2 style='color: #FFFFFF; margin-bottom: 0.2rem;'>Enterprise Intelligence Console</h2>",
            unsafe_allow_html=True)
st.caption(
    "Query internal documentation through multi-step agent reasoning, contextual retrieval, and grounded synthesis.")
st.markdown("<br>", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("metadata"):
            meta = msg["metadata"]
            retrieval_badge = (
                "<span class='status-badge badge-green'>Knowledge Base Route</span>"
                if meta["used_retrieval"]
                else "<span class='status-badge badge-gray'>Direct Route</span>"
            )
            rewrite_badge = (
                f"<span class='status-badge badge-gray'>Rewrites: {meta['rewritten_iterations']}</span>"
                if meta["rewritten_iterations"] > 0
                else ""
            )

            st.markdown(f"<div style='margin-top: 8px; margin-bottom: 8px;'>{retrieval_badge} {rewrite_badge}</div>",
                        unsafe_allow_html=True)

            if meta.get("sources"):
                with st.expander(f"Cited Evidence ({len(meta['sources'])} segments)"):
                    for idx, src in enumerate(meta["sources"]):
                        score = f"{src['relevance_score']:.4f}" if src.get("relevance_score") is not None else "N/A"
                        page = src.get("metadata", {}).get("page", "N/A")
                        doc_name = src.get("metadata", {}).get("source", "Knowledge Base")

                        st.markdown(f"**Segment [{idx + 1}]** | {doc_name} | Page {page} | Score: `{score}`")
                        st.caption(src["content"])
                        if idx < len(meta["sources"]) - 1:
                            st.divider()

# Input Bar
if prompt := st.chat_input("Enter your technical inquiry..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Executing retrieval and agent verification..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/query",
                    json={"query": prompt},
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    answer_text = result["response"]
                    sources = result.get("sources", [])
                    rewritten_iterations = result.get("rewritten_iterations", 0)
                    used_retrieval = result.get("used_retrieval", False)

                    st.markdown(answer_text)

                    meta = {
                        "sources": sources,
                        "rewritten_iterations": rewritten_iterations,
                        "used_retrieval": used_retrieval
                    }

                    retrieval_badge = (
                        "<span class='status-badge badge-green'>Knowledge Base Route</span>"
                        if used_retrieval
                        else "<span class='status-badge badge-gray'>Direct Route</span>"
                    )
                    rewrite_badge = (
                        f"<span class='status-badge badge-gray'>Rewrites: {rewritten_iterations}</span>"
                        if rewritten_iterations > 0
                        else ""
                    )

                    st.markdown(
                        f"<div style='margin-top: 8px; margin-bottom: 8px;'>{retrieval_badge} {rewrite_badge}</div>",
                        unsafe_allow_html=True)

                    if used_retrieval and sources:
                        with st.expander(f"Cited Evidence ({len(sources)} segments)"):
                            for idx, src in enumerate(sources):
                                score = f"{src['relevance_score']:.4f}" if src.get(
                                    "relevance_score") is not None else "N/A"
                                page = src.get("metadata", {}).get("page", "N/A")
                                doc_name = src.get("metadata", {}).get("source", "Knowledge Base")

                                st.markdown(f"**Segment [{idx + 1}]** | {doc_name} | Page {page} | Score: `{score}`")
                                st.caption(src["content"])
                                if idx < len(sources) - 1:
                                    st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "metadata": meta
                    })
                else:
                    st.error(f"Query failed with status code {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Connection refused: Ensure the FastAPI backend is running on port 8000.")
            except Exception as e:
                st.error(f"Execution error: {e}")