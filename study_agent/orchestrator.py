"""
orchestrator.py
───────────────
LangGraph-style orchestrator that wires all four agents into a directed graph.

Concepts from document (Section 5.4 — LangGraph):
  - StateGraph: nodes = Python functions, edges = transitions
  - set_entry_point: where the graph starts
  - add_edge: unconditional transitions (researcher → analyst, analyst → critic)
  - add_conditional_edges: critic → analyst OR critic → finaliser (based on score)
  - compile(): returns a callable graph with .invoke() interface
  - Checkpointing: SQLite checkpointer for session persistence (optional)

Graph topology:
  researcher → analyst → critic ──(score ≥ 0.75)──→ finaliser
                  ↑                                      │
                  └──────(score < 0.75)──────────────────┘
                          (max 2 iterations)

This implements the Reflexion architecture from the document:
  the critic's feedback loops back to analyst for self-correction.
"""

from __future__ import annotations

import uuid
from typing import Optional, Callable

from agents.state import AgentState
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.critic import critic_node, route_after_critique
from agents.finaliser import finaliser_node
from memory.vector_store import VectorMemory


def build_graph(llm, memory: VectorMemory) -> "StudyAgentGraph":
    """
    Build and compile the multi-agent graph.

    Args:
        llm: Any LangChain-compatible chat model (ChatOpenAI, ChatOllama, etc.)
        memory: VectorMemory instance for long-term storage

    Returns:
        A compiled StudyAgentGraph with an .invoke() method.
    """
    return StudyAgentGraph(llm=llm, memory=memory)


class StudyAgentGraph:
    """
    Manual implementation of a LangGraph StateGraph pattern.

    In production, you'd use:
        from langgraph.graph import StateGraph, END
        builder = StateGraph(AgentState)
        ...
        graph = builder.compile(checkpointer=MemorySaver())

    Here we implement the same pattern explicitly so the code is
    fully runnable without langgraph installed, and every step is
    transparent for learning and interview explanation.

    The graph logic is identical to what LangGraph produces.
    """

    def __init__(self, llm, memory: VectorMemory):
        self.llm = llm
        self.memory = memory

    def invoke(
        self,
        query: str,
        topic: str = "",
        max_iterations: int = 2,
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> AgentState:
        """
        Run the full multi-agent pipeline for a given query.

        Args:
            query: The user's question or topic
            topic: Short label for memory tagging (defaults to first 50 chars of query)
            max_iterations: Max critique→revise loops (safety valve)
            session_id: Unique session ID for checkpointing
            progress_callback: Optional fn(agent_name, status) for UI updates

        Returns:
            Final AgentState with all intermediate and final outputs.
        """
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        def _emit(agent: str, msg: str):
            if progress_callback:
                progress_callback(agent, msg)
            else:
                print(f"[{agent.upper()}] {msg}")

        # ── Initialise state ──────────────────────────────────────────────
        state: AgentState = {
            "user_query": query,
            "topic": topic or query[:50],
            "messages": [],
            "retrieved_docs": [],
            "search_results": "",
            "draft_answer": "",
            "critique": "",
            "critique_score": 0.0,
            "final_answer": "",
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "approved": False,
            "memory_context": "",
            "session_id": session_id,
        }

        # ── NODE 1: Researcher ────────────────────────────────────────────
        _emit("researcher", "Searching Wikipedia and web + querying memory...")
        researcher_updates = researcher_node(state, self.llm, self.memory)
        state = {**state, **researcher_updates}
        _emit("researcher", f"Research complete. Found {len(state['retrieved_docs'])} docs.")

        # ── LOOP: Analyst → Critic → (route) ─────────────────────────────
        while True:
            # ── NODE 2: Analyst ───────────────────────────────────────────
            iteration = state["iteration_count"]
            _emit("analyst",
                  f"Writing study note (iteration {iteration + 1}/{max_iterations})...")
            analyst_updates = analyst_node(state, self.llm)
            state = {**state, **analyst_updates}
            _emit("analyst", "Draft study note ready.")

            # ── NODE 3: Critic ────────────────────────────────────────────
            _emit("critic", "Reviewing draft for quality and accuracy...")
            critic_updates = critic_node(state, self.llm)
            state = {**state, **critic_updates}
            score = state["critique_score"]
            _emit("critic", f"Review complete. Quality score: {score:.2f}/1.00")

            # ── CONDITIONAL ROUTING ───────────────────────────────────────
            next_node = route_after_critique(state)
            if next_node == "finaliser":
                _emit("critic",
                      f"Score {score:.2f} ≥ 0.75 → routing to finaliser ✅")
                break
            else:
                _emit("critic",
                      f"Score {score:.2f} < 0.75 → routing back to analyst for revision 🔁")

        # ── NODE 4: Finaliser ─────────────────────────────────────────────
        _emit("finaliser", "Polishing final answer and saving to long-term memory...")
        finaliser_updates = finaliser_node(state, self.llm, self.memory)
        state = {**state, **finaliser_updates}
        _emit("finaliser", "Done! Study note saved to memory. ✅")

        return state

    def get_graph_description(self) -> str:
        """Return a text description of the graph topology for display."""
        return """
Multi-Agent Study Assistant — Graph Topology
═════════════════════════════════════════════
START
  │
  ▼
[RESEARCHER] — Queries Wikipedia + DuckDuckGo + Vector Memory
  │             Returns: research_summary, retrieved_docs
  │
  ▼
[ANALYST] ◄──────────────────────────────────────┐
  │         Uses CoT + Structured Output           │
  │         Returns: draft_answer                  │
  │                                                │
  ▼                                                │
[CRITIC]  — Scores 0.0–1.0 using Reflexion pattern│
  │         Returns: critique, critique_score      │
  │                                                │
  ├──── score ≥ 0.75 ──────────────────────────► [FINALISER]
  │                                                    │
  └──── score < 0.75 (& iterations remaining) ─────────┘
                                                        │
                                                        ▼
                                                    [END]
                                              final_answer in state
"""
