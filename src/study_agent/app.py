"""
app.py
──────
Streamlit web UI for the Multi-Agent Study Assistant.

Run with: streamlit run app.py

Features:
  - Real-time agent progress updates using st.status()
  - Study note displayed as formatted Markdown
  - Memory browser: view all past stored notes
  - Agent pipeline diagram embedded in sidebar
  - Query examples for quick demos
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Study Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.agent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px;
}
.badge-researcher { background: #dbeafe; color: #1e40af; }
.badge-analyst    { background: #dcfce7; color: #166534; }
.badge-critic     { background: #fef3c7; color: #92400e; }
.badge-finaliser  { background: #f3e8ff; color: #6b21a8; }
.score-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []
if "memory_initialized" not in st.session_state:
    st.session_state.memory_initialized = False


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        ["openai", "anthropic", "ollama"],
        help="OpenAI recommended. Ollama is free but requires local setup."
    )

    model_map = {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "ollama": ["llama3", "mistral", "phi3", "gemma2"],
    }
    model = st.selectbox("Model", model_map[provider])

    max_iter = st.slider(
        "Max Critic→Analyst Iterations",
        min_value=1, max_value=3, value=2,
        help="Safety valve: max times Critic can send back to Analyst"
    )

    quality_threshold = st.slider(
        "Quality Threshold",
        min_value=0.5, max_value=0.95, value=0.75, step=0.05,
        help="Score needed to pass from Critic to Finaliser"
    )

    st.divider()
    st.subheader("🗺️ Agent Graph")
    st.code("""
researcher
    │
    ▼
analyst ◄────────┐
    │             │ (score < 0.75)
    ▼             │
 critic  ─────────┘
    │
    │ (score ≥ 0.75)
    ▼
finaliser
    │
   END
""", language=None)

    st.divider()
    st.subheader("📚 Concepts Shown")
    concepts = [
        "ReAct Pattern", "RAG Pipeline", "Chain-of-Thought",
        "Reflexion (Critic)", "Long-term Memory", "ChromaDB",
        "Sentence Embeddings", "Multi-Agent Orchestration",
        "Conditional Routing", "Tool Use", "Structured Output",
        "Few-shot Prompting", "Role/Persona Prompting",
    ]
    for c in concepts:
        st.markdown(f"✅ {c}")


# ── Main content ─────────────────────────────────────────────────────────────
st.title("🤖 Multi-Agent Study Assistant")
st.caption("GenAI + Agentic AI · RAG + Multi-Agent Orchestration · LangGraph Patterns")

# ── Example queries ───────────────────────────────────────────────────────────
st.subheader("💡 Quick Start — Example Queries")
examples = [
    "Explain transformer self-attention mechanism",
    "What is RAG and how does it reduce hallucination?",
    "Explain the ReAct agent pattern with an example",
    "What is cosine similarity and why is it used for embeddings?",
    "How does LangGraph StateGraph work with nodes and conditional edges?",
    "Explain the difference between Chain-of-Thought and Plan-and-Execute",
]

cols = st.columns(3)
selected_example = None
for i, ex in enumerate(examples):
    if cols[i % 3].button(f"📌 {ex[:45]}...", key=f"ex_{i}", use_container_width=True):
        selected_example = ex

# ── Query input ───────────────────────────────────────────────────────────────
st.divider()
query = st.text_area(
    "📝 Your Study Query",
    value=selected_example or "",
    placeholder="e.g. Explain how HNSW indexing works in vector databases",
    height=80,
)
topic = st.text_input("🏷️ Topic Label (optional, for memory tagging)", placeholder="e.g. Vector Databases")

run_button = st.button("🚀 Run Multi-Agent Pipeline", type="primary", use_container_width=True)

# ── Run the pipeline ──────────────────────────────────────────────────────────
if run_button and query.strip():
    st.divider()
    st.subheader("⚙️ Agent Pipeline Running...")

    # Import here to avoid startup errors if deps missing
    try:
        from study_agent.llm_factory import get_llm
        from study_agent.memory.vector_store import VectorMemory
        from study_agent.orchestrator import build_graph
    except Exception as e:
        st.error(f"Import error: {e}\n\nRun: pip install -r requirements.txt")
        st.stop()

    # Set env var for provider selection
    os.environ["__STUDY_AGENT_PROVIDER"] = provider

    # Monkeypatch the auto-detect to use UI selection
    import study_agent.llm_factory as _lf
    _orig = _lf._auto_detect_provider
    _lf._auto_detect_provider = lambda: provider

    try:
        llm = get_llm(provider=provider, model=model)
    except Exception as e:
        st.error(f"LLM init failed: {e}")
        _lf._auto_detect_provider = _orig
        st.stop()
    finally:
        _lf._auto_detect_provider = _orig

    memory = VectorMemory()
    graph = build_graph(llm=llm, memory=memory)

    # ── Progress display ──────────────────────────────────────────────────
    progress_container = st.container()
    steps_log = []

    agent_styles = {
        "researcher": ("🔍", "badge-researcher", "Searching Wikipedia + Web + Memory"),
        "analyst":    ("✍️", "badge-analyst",    "Writing structured study note (CoT)"),
        "critic":     ("🔎", "badge-critic",     "Reviewing with Reflexion pattern"),
        "finaliser":  ("✨", "badge-finaliser",  "Polishing + saving to ChromaDB"),
    }

    step_placeholders = {}
    with progress_container:
        for agent in ["researcher", "analyst", "critic", "finaliser"]:
            emoji, badge, desc = agent_styles[agent]
            step_placeholders[agent] = st.empty()
            step_placeholders[agent].markdown(
                f'<span class="agent-badge {badge}">⏳ {emoji} {agent.upper()}</span> '
                f'<span style="color:#888">{desc}</span>',
                unsafe_allow_html=True
            )

    result_placeholder = st.empty()
    log_placeholder = st.empty()

    def progress_callback(agent: str, status: str):
        steps_log.append((agent, status))
        emoji, badge, _ = agent_styles.get(agent.lower(), ("⚙️", "", ""))

        # Update agent step display
        if agent.lower() in step_placeholders:
            step_placeholders[agent.lower()].markdown(
                f'<span class="agent-badge {badge}">✅ {emoji} {agent.upper()}</span> '
                f'<span>{status}</span>',
                unsafe_allow_html=True
            )

    # ── Execute pipeline ──────────────────────────────────────────────────
    with st.spinner("Multi-agent pipeline running..."):
        try:
            result = graph.invoke(
                query=query.strip(),
                topic=topic.strip() or query.strip()[:50],
                max_iterations=max_iter,
                progress_callback=progress_callback,
            )
            st.session_state.result = result
            st.session_state.history.append({
                "query": query,
                "result": result,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    # ── Display results ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Final Study Note")

    # Score display
    score = result.get("critique_score", 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Quality Score", f"{score:.2f}/1.00",
                delta="Passed ✅" if score >= 0.75 else "Auto-approved")
    col2.metric("Iterations", result.get("iteration_count", 1))
    col3.metric("Memory Notes", memory.count())

    # Final answer
    st.markdown(result["final_answer"])

    # Expandable: intermediate outputs
    with st.expander("🔍 Researcher Output (RAG + Search)", expanded=False):
        docs = result.get("retrieved_docs", [])
        if docs:
            st.markdown(docs[0] if docs else "No research retrieved.")
        st.markdown("**Memory Context Retrieved:**")
        st.markdown(result.get("memory_context", "_No past notes retrieved_") or "_No past notes retrieved_")

    with st.expander("✍️ Analyst Draft (before critique)", expanded=False):
        st.markdown(result.get("draft_answer", "N/A"))

    with st.expander("🔎 Critic's Review", expanded=False):
        st.markdown(result.get("critique", "N/A"))

    # Download button
    st.download_button(
        label="⬇️ Download Study Note (.md)",
        data=result["final_answer"],
        file_name=f"study_note_{topic or 'query'}.md",
        mime="text/markdown",
    )

elif run_button and not query.strip():
    st.warning("Please enter a study query first.")


# ── Memory Browser ────────────────────────────────────────────────────────────
st.divider()
st.subheader("🧠 Long-Term Memory Browser")
st.caption("Notes saved to ChromaDB — persisted across sessions, retrieved by semantic similarity")

try:
    from study_agent.memory.vector_store import VectorMemory
    mem = VectorMemory()
    topics = mem.list_topics()
    count = mem.count()

    if count > 0:
        st.success(f"📚 {count} study notes stored in ChromaDB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Topics in memory:**")
            for t in sorted(topics):
                st.markdown(f"• {t}")
        with col2:
            search_mem = st.text_input("🔍 Search memory semantically:", key="mem_search")
            if search_mem:
                results = mem.retrieve(search_mem, n_results=3)
                if results:
                    for r in results:
                        with st.expander(f"📄 {r['topic']} (similarity: {r.get('similarity', 0):.2f})"):
                            st.markdown(r["content"][:500] + "...")
                else:
                    st.info("No relevant notes found for that query.")
    else:
        st.info("No study notes in memory yet. Run a query to start building your knowledge base!")
except Exception as e:
    st.warning(f"Memory browser unavailable: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Architecture**: ReAct · Chain-of-Thought · Reflexion · RAG · "
    "ChromaDB · sentence-transformers · LangGraph patterns · "
    "Multi-Agent Orchestration (Researcher → Analyst → Critic → Finaliser)"
)
