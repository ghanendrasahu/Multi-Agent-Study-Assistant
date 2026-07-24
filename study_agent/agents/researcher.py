"""
agents/researcher.py
────────────────────
RESEARCHER AGENT — Phase 1 of the pipeline

Role      : Gather relevant knowledge for the user's query
Pattern   : ReAct (Reason + Act) — decides which tool to call
Tools     : Wikipedia search, long-term vector memory
Prompt    : Role/Persona prompting + Zero-shot ReAct instruction

Concepts from document:
  - ReAct pattern: the agent outputs Thought → Action → Observation in a loop
  - Tool use: agent decides which tool to call based on the query
  - RAG retrieval: pulls semantically similar past notes from ChromaDB
  - Role prompting: "You are an expert research assistant..."
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from memory.vector_store import VectorMemory
from tools.search import wikipedia_search, duckduckgo_search


# ── System prompt (Role/Persona prompting pattern) ─────────────────────────
RESEARCHER_SYSTEM_PROMPT = """You are an expert research assistant specialising in \
Data Science, AI, Machine Learning, and Computer Science.

Your job is to gather comprehensive, accurate information on the given topic. \
You have access to:
1. Wikipedia (for factual, well-sourced overviews)
2. DuckDuckGo web search (for recent developments)
3. The user's own past study notes (long-term memory)

Think step-by-step about what information would be most useful, then retrieve it.
Return a well-structured research summary — no bullet points, flowing paragraphs.
Focus on: core concepts, how it works, why it matters, key terminology."""


def researcher_node(state: AgentState, llm, memory: VectorMemory) -> dict:
    """
    Researcher node: retrieves knowledge and builds a research summary.

    ReAct loop (simplified):
        Thought  → What do I need to know?
        Action   → Call wikipedia_search / duckduckgo_search
        Observe  → Read results
        Thought  → What from memory is relevant?
        Action   → Query vector store
        Observe  → Read past notes
        Respond  → Synthesise into research summary
    """
    query = state["user_query"]

    # ── Step 1: Query long-term memory (RAG retrieval) ────────────────────
    past_notes = memory.retrieve(query, n_results=3)
    memory_context = ""
    if past_notes:
        memory_context = "=== RELEVANT PAST STUDY NOTES ===\n"
        for note in past_notes:
            memory_context += f"\n[Topic: {note['topic']}]\n{note['content']}\n"
        memory_context += "\n=================================\n"

    # ── Step 2: Wikipedia search (primary tool) ───────────────────────────
    wiki_result = wikipedia_search(query)

    # ── Step 3: Web search for supplementary info ─────────────────────────
    web_result = duckduckgo_search(query, max_results=3)

    # ── Step 4: Build combined context ────────────────────────────────────
    search_context = f"""
WIKIPEDIA RESULT:
{wiki_result}

WEB SEARCH RESULTS:
{web_result}
"""

    # ── Step 5: LLM synthesises into a research summary ───────────────────
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"""
{memory_context}

SEARCH RESULTS TO SYNTHESISE:
{search_context}

USER QUERY: {query}

Please synthesise the above into a comprehensive research summary.
Structure it as:
1. Core Concept
2. How It Works
3. Key Terminology
4. Why It Matters
5. Connections to Related Concepts
""")
    ]

    response = llm.invoke(messages)
    research_summary = response.content

    return {
        "search_results": search_context,
        "retrieved_docs": [research_summary],
        "memory_context": memory_context,
        "messages": [
            HumanMessage(content=f"Research query: {query}"),
            AIMessage(content=research_summary, name="researcher")
        ]
    }
