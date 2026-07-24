"""
agents/finaliser.py
───────────────────
FINALISER AGENT — Phase 4 (final phase) of the pipeline

Role      : Polish the approved draft, add metadata, save to long-term memory
Pattern   : Memory consolidation — writes to ChromaDB for future retrieval
Output    : Final formatted study note + confirmation of memory storage

Concepts from document:
  - Long-term memory write: episodic memory stored in vector DB
  - Memory consolidation strategy: only high-quality (approved) notes saved
  - Output formatting: ensures consistent Markdown structure
  - Minimal footprint principle: agent writes ONLY when quality is assured
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from study_agent.agents.state import AgentState
from study_agent.memory.vector_store import VectorMemory


FINALISER_SYSTEM_PROMPT = """You are the final editor of an AI study assistant. \
Your job is to take an approved study note and make it perfect.

Tasks:
1. Add a "📊 Quality Score" line at the top showing the critic's score
2. Add a "🔁 Session Summary" section at the very end with 3 bullet points: \
   the key insight, the best analogy, and one exam question on this topic
3. Ensure all formatting is clean and consistent
4. Do NOT change the content significantly — just polish and augment

The output should be ready to paste into Notion, Obsidian, or a study document."""


def finaliser_node(state: AgentState, llm, memory: VectorMemory) -> dict:
    """
    Finaliser node:
      1. Polish the approved draft
      2. Save it to long-term vector memory (ChromaDB)
      3. Return the final formatted output
    """
    draft = state.get("draft_answer", "")
    score = state.get("critique_score", 0.75)
    query = state["user_query"]
    topic = state.get("topic", query[:50])
    session_id = state.get("session_id", "default")

    messages = [
        SystemMessage(content=FINALISER_SYSTEM_PROMPT),
        HumanMessage(content=f"""
APPROVED STUDY NOTE (quality score: {score:.2f}):
{draft}

ORIGINAL QUERY: {query}

Please finalise this note. Add the quality score header and session summary section.
""")
    ]

    response = llm.invoke(messages)
    final_answer = response.content

    # ── Save to long-term memory (ChromaDB) ───────────────────────────────
    memory_saved = False
    try:
        memory.store(
            content=final_answer,
            topic=topic,
            query=query,
            score=score,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        memory_saved = True
    except Exception as e:
        # Non-fatal: memory write failure should not crash the agent
        print(f"[WARNING] Memory write failed: {e}")

    # ── Append memory confirmation footer ─────────────────────────────────
    memory_footer = (
        "\n\n---\n✅ **Saved to long-term study memory** — "
        "this note will be retrieved in future sessions on related topics."
        if memory_saved
        else "\n\n---\n⚠️ Memory save skipped (storage unavailable)"
    )

    final_with_footer = final_answer + memory_footer

    return {
        "final_answer": final_with_footer,
        "messages": [AIMessage(content=final_with_footer, name="finaliser")]
    }
