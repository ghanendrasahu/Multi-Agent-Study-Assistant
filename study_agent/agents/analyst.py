"""
agents/analyst.py
─────────────────
ANALYST AGENT — Phase 2 of the pipeline

Role      : Convert raw research into a structured, student-friendly study note
Pattern   : Chain-of-Thought (CoT) + Structured Output (Pydantic schema)
Prompt    : Few-shot + explicit output format instruction

Concepts from document:
  - Chain-of-Thought: "Think step by step before writing the final answer"
  - Structured output: LLM is prompted to return a specific JSON/Markdown schema
  - Few-shot prompting: one example study note is provided in the system prompt
  - Output format control: explicit schema in the prompt guarantees parseable output
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState


# ── Few-shot example embedded in system prompt ─────────────────────────────
FEW_SHOT_EXAMPLE = """
EXAMPLE STUDY NOTE (for topic: "Cosine Similarity"):

## 📚 Cosine Similarity

### 🎯 One-Line Definition
A metric that measures the angle between two vectors in high-dimensional space, \
returning a value between -1 and 1.

### 🔍 Core Concept
Cosine similarity focuses on *direction* rather than *magnitude*. Two vectors \
pointing in the same direction score 1.0, perpendicular vectors score 0.0, and \
opposing vectors score -1.0. This makes it ideal for text embeddings where the \
length of a document shouldn't affect similarity.

### 🛠️ How It Works (Step-by-Step)
1. Convert each text into an embedding vector
2. Compute the dot product of the two vectors
3. Divide by the product of their magnitudes
4. Result: cos(θ) = (A·B) / (‖A‖ × ‖B‖)

### 💡 Simple Analogy
Think of two people walking in a city. Cosine similarity measures whether they're \
walking in the *same direction*, not how far they've walked.

### 🔑 Key Terms
- **Dot Product**: sum of element-wise multiplication of two vectors
- **Magnitude**: the Euclidean length of a vector (‖v‖ = √(Σvᵢ²))
- **Angle θ**: the angle between vectors; smaller angle = higher similarity

### ⚠️ Common Misconceptions
- Does NOT measure absolute distance — two short and long texts on the same \
  topic score high
- Score of 0 means *unrelated*, not *opposite*

### 🔗 Connected Concepts
Embeddings → Vector Databases → Semantic Search → RAG Retrieval
"""


ANALYST_SYSTEM_PROMPT = f"""You are an expert AI/ML educator who specialises in \
creating clear, structured study notes for students.

Your job is to take research material and transform it into a well-structured \
study note that a student can use for revision.

Always think step-by-step (Chain-of-Thought) before writing your final answer.
Follow the EXACT structure shown in the example below.

{FEW_SHOT_EXAMPLE}

Rules:
- Use simple analogies to explain complex concepts
- Always include a "Connected Concepts" section showing how topics link together
- Highlight common misconceptions — these are gold for exam preparation
- Use emoji headers exactly as shown in the example
- Be concise but complete — each section should be 2-5 sentences
"""


def analyst_node(state: AgentState, llm) -> dict:
    """
    Analyst node: converts research into a structured study note.

    Uses CoT prompting:
      1. First, the LLM is instructed to reason about the topic
      2. Then it fills the structured template
    """
    # Combine all retrieved research
    research = "\n\n".join(state.get("retrieved_docs", []))
    query = state["user_query"]

    # If this is a re-analysis (after critique), inject the critique feedback
    critique = state.get("critique", "")
    iteration = state.get("iteration_count", 0)

    critique_injection = ""
    if critique and iteration > 0:
        critique_injection = f"""
⚠️ IMPORTANT — This is revision #{iteration}. The previous draft was critiqued:

CRITIQUE FEEDBACK:
{critique}

Address ALL points raised above in your revised study note.
"""

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=f"""
RESEARCH MATERIAL:
{research}

USER QUERY / TOPIC: {query}

{critique_injection}

Think step by step:
1. What is the single most important concept here?
2. What analogy best explains it?
3. What do students commonly get wrong?
4. What other topics connect to this?

Now write the structured study note following the template exactly.
""")
    ]

    response = llm.invoke(messages)
    draft_answer = response.content

    return {
        "draft_answer": draft_answer,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [
            AIMessage(content=draft_answer, name="analyst")
        ]
    }
