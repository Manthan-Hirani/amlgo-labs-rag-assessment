"""
RAG Chatbot - Streamlit Application
====================================
AI-powered chatbot with streaming responses for answering
questions based on indexed documents.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

from src.pipeline import RAGPipeline
from src.embeddings import get_collection_count, EMBEDDING_MODEL
from src.generator import DEFAULT_MODEL


# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot | Amlgo Labs",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Dark Theme ────────────────────────────────
st.markdown("""
<style>
    /* Main container layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* Header alignment */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1rem;
    }

    .app-header h1 {
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }

    .app-header p {
        font-size: 0.95rem;
        opacity: 0.8;
    }

    /* Source card structure */
    .source-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .score-badge {
        background-color: rgba(128, 128, 128, 0.2);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .source-text {
        font-size: 0.85rem;
        line-height: 1.6;
        opacity: 0.9;
    }

    /* Sidebar sections structure */
    .sidebar-section {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
    }

    .sidebar-section h3 {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }

    .metric-item {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        font-size: 0.85rem;
    }

    .metric-value {
        font-weight: 500;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        background: #4ade80;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Initialize Pipeline ──────────────────────────────────────────────
@st.cache_resource
def init_pipeline():
    """Initialize the RAG pipeline (cached across reruns)."""
    project_root = Path(__file__).parent
    vectordb_dir = project_root / "vectordb"
    return RAGPipeline(vectordb_dir=str(vectordb_dir))


# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="font-size: 1.4rem; font-weight: bold;">🤖 RAG Chatbot</h2>
        <p style="font-size: 0.85rem; opacity: 0.8;">Powered by Ollama & ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)

    # System info section
    project_root = Path(__file__).parent
    chunk_count = get_collection_count(str(project_root / "vectordb"))

    st.markdown(f"""
    <div class="sidebar-section">
        <h3>System Info</h3>
        <div class="metric-item">
            <span class="metric-label">LLM Model</span>
            <span class="metric-value">{DEFAULT_MODEL}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Embedding</span>
            <span class="metric-value">{EMBEDDING_MODEL}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Indexed Chunks</span>
            <span class="metric-value">{chunk_count}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Vector DB</span>
            <span class="metric-value">ChromaDB</span>
        </div>
        <div class="metric-item" style="border-bottom: none;">
            <span class="metric-label">Status</span>
            <span class="metric-value"><span class="status-dot status-online"></span>Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Retrieval settings
    st.markdown("##### Retrieval Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 15, 5, key="top_k_slider")

    st.markdown("---")

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Sample queries
    st.markdown("##### Sample Queries")
    sample_queries = [
        "Which specific eBay entity am I contracting with if I reside in the United Kingdom?",
        "Under the Agreement to Arbitrate, what is the mandatory precondition before commencing arbitration, and exactly how long does it last?",
        "How do I change the oil and replace the filters on a 2024 Ford F-150?",
        "The document says eBay guarantees 100% accuracy for its AI tools. Can you explain how they achieve this?",
        "I am a 16-year-old living in the United States. Which eBay entity do I contract with?",
        "Can you explain the specific $50 flat fee for selling a used smartphone mentioned in Section 6?",
        "Write a 500-word romantic poem about the Agreement to Arbitrate in Section 19.",
        "In your opinion, is the 'Agreement to Arbitrate' fair for the average consumer, or is it designed just to protect eBay's profits?",
    ]
    for i, q in enumerate(sample_queries):
        # Use a tool-tip (help) for the full query and a slightly truncated label if needed, 
        # but for now we just use the query as the button label.
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()


# ─── Main Chat Area ──────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Chat Assistant</h1>
    <p>Ask questions about the indexed documents — powered by RAG with real-time streaming</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Source Passages Used", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    score_pct = f"{source['relevance_score'] * 100:.1f}%"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span> Source {i} | Page {source['source_page']}</span>
                            <span class="score-badge">Relevance: {score_pct}</span>
                        </div>
                        <div class="source-text">{source['text'][:300]}{'...' if len(source['text']) > 300 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Handle pending query from sidebar buttons
pending = st.session_state.pop("pending_query", None)

# Chat input
user_input = st.chat_input("Ask a question about the documents...")

# Use pending query or typed input
query = pending or user_input

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        try:
            pipeline = init_pipeline()
            pipeline.top_k = top_k

            # Extract conversation history for multi-turn support
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # Exclude current query
                if m["role"] in ("user", "assistant")
            ]

            # Get streaming response and sources with history
            token_stream, sources = pipeline.query_stream(
                query, conversation_history=history if history else None
            )

            # Pre-add to history for partial saving in case of interrupt
            message_idx = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "sources": sources,
            })

            def stream_wrapper(gen):
                for token in gen:
                    st.session_state.messages[message_idx]["content"] += token
                    yield token

            text_container = st.container()
            stop_placeholder = st.empty()
            
            with stop_placeholder:
                # Button triggers a rerun, which interrupts the stream and frees the input
                st.button("Stop Generating", key=f"stop_btn_{message_idx}")

            with text_container:
                response = st.write_stream(stream_wrapper(token_stream))
            
            # Clear stop button when done
            stop_placeholder.empty()

            # Ensure final response is exactly what write_stream returned
            st.session_state.messages[message_idx]["content"] = response

            # Display sources
            with st.expander("Source Passages Used", expanded=False):
                for i, source in enumerate(sources, 1):
                    score_pct = f"{source['relevance_score'] * 100:.1f}%"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span> Source {i} | Page {source['source_page']}</span>
                            <span class="score-badge">Relevance: {score_pct}</span>
                        </div>
                        <div class="source-text">{source['text'][:300]}{'...' if len(source['text']) > 300 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
