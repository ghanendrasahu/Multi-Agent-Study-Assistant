"""
agents/critic.py
────────────────
CRITIC AGENT — Phase 3 of the pipeline (with conditional routing)

Role      : Evaluate the analyst's study note and decide: approve OR send back
Pattern   : Reflexion — self-reflection + memory-augmented revision signal
Routing   : Returns score; orchestrator routes to finaliser (≥0.75) OR analyst (<0.75)

Concepts from document:
  - Reflexion cognitive architecture: agents reflect on their own outputs
  - Confidence estimation: explicit 0-1 score used as a decision gate
  - Conditional edges in LangGraph: score determines next node
  - Self-correction: if score low, critique is injected into analyst on next pass
  - Iteration safety valve: max_iterations prevents infinite loops
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from study_agent.agents.state import AgentState


CRITIC_SYSTEM_PROMPT = """You are a strict but fair academic reviewer evaluating \
AI/ML study notes for a student who will use them in interviews and exams.

Your task:
1. Read the study note carefully
2. Score it on each dimension (0.0 to 1.0)
3. Give specific, actionable feedback
4. Output an overall SCORE between 0.0 and 1.0

Evaluation dimensions:
- **Accuracy** (0-1): Is the content factually correct? No hallucinations?
- **Clarity** (0-1): Is it easy to understand for a student?
- **Completeness** (0-1): Are key concepts, terms, and examples covered?
- **Structure** (0-1): Does it follow the required template?
- **Analogy Quality** (0-1): Is the analogy apt and memorable?

CRITICAL OUTPUT FORMAT — You MUST end your response with exactly:
SCORE: X.XX

Where X.XX is your overall average score (e.g. SCORE: 0.82).
If score < 0.75, the note will be sent back for revision — make your feedback specific.
"""


def critic_node(state: AgentState, llm) -> dict:
    """
    Critic node: reviews the draft answer and scores it.

    Key patterns:
      - Reflexion: explicitly critiques and identifies gaps
      - Confidence gate: score < 0.75 triggers a re-analysis loop
      - Safety valve: if max_iterations reached, always approve (avoids infinite loop)
    """
    draft = state.get("draft_answer", "")
    query = state["user_query"]
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 2)

    # ── Safety valve: force approval if max iterations reached ────────────
    if iteration >= max_iter:
        return {
            "critique": "Max iterations reached — auto-approving to prevent infinite loop.",
            "critique_score": 0.75,  # Force passage
            "messages": [AIMessage(
                content="[Auto-approved after reaching max iterations]",
                name="critic"
            )]
        }

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"""
ORIGINAL USER QUERY: {query}

STUDY NOTE TO REVIEW:
{draft}

Please evaluate the above study note. Be specific about what's missing or wrong.
Remember to end with: SCORE: X.XX
""")
    ]

    response = llm.invoke(messages)
    critique_text = response.content

    # ── Parse the score from the response ─────────────────────────────────
    score = _extract_score(critique_text)

    return {
        "critique": critique_text,
        "critique_score": score,
        "messages": [AIMessage(content=critique_text, name="critic")]
    }


def route_after_critique(state: AgentState) -> str:
    """
    Conditional routing function — called by the graph after the critic node.

    Returns:
      "analyst"   → score < 0.75 AND iterations remaining → loop back
      "finaliser" → score ≥ 0.75 OR max iterations reached → finish
    """
    score = state.get("critique_score", 0.0)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 2)

    if score >= 0.75 or iteration >= max_iter:
        return "finaliser"
    else:
        return "analyst"


def _extract_score(text: str) -> float:
    """Parse 'SCORE: 0.82' from critic response. Defaults to 0.75 if not found."""
    match = re.search(r"SCORE:\s*([01]\.\d+)", text, re.IGNORECASE)
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    # If LLM didn't follow format, be generous and approve
    return 0.75
