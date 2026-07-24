"""
agents/state.py
───────────────
Shared TypedDict state that flows through every node in the agent graph.

Key LangGraph pattern:
  - Fields annotated with `Annotated[list, operator.add]` ACCUMULATE across nodes
    (messages, retrieved_docs).
  - Plain fields are OVERWRITTEN by the last node that writes them.

This is the most critical design decision in a LangGraph workflow.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    user_query: str                  # Original question from the user
    topic: str                       # Short topic label (for memory tagging)

    # ── Accumulating fields (operator.add = append, not overwrite) ─────────
    messages: Annotated[list, operator.add]       # Full message history
    retrieved_docs: Annotated[list, operator.add] # Docs fetched by researcher

    # ── Pipeline fields (overwritten each step) ────────────────────────────
    search_results: str              # Raw search results from tools
    draft_answer: str                # Analyst's first-pass answer
    critique: str                    # Critic's structured critique
    critique_score: float            # 0.0 – 1.0 quality score from critic
    final_answer: str                # Finaliser's polished output

    # ── Control flow ───────────────────────────────────────────────────────
    iteration_count: int             # Safety valve against infinite loops
    max_iterations: int              # Configurable ceiling (default: 2)
    approved: bool                   # Human-in-the-loop approval flag

    # ── Memory ─────────────────────────────────────────────────────────────
    memory_context: str              # Relevant past notes injected at start
    session_id: str                  # Unique session identifier
