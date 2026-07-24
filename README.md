# 🤖 Multi-Agent Study Assistant

> **Try it live:** [![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ghanendrasahu-multi-agent-study-assistant-app-ghxuhm.streamlit.app/)
>
> Or clone & run locally — instructions below.

A production-style **Agentic AI** application that demonstrates the full GenAI stack:
**LLM → Embeddings → RAG → Agent Loop → Multi-Agent Orchestration → Memory**

---

## 🎯 What It Does

You give it any topic (e.g. *"Explain transformer self-attention"*) and a **pipeline of 4 specialized AI agents** collaborates to produce:

- ✅ A structured study note (concept, examples, analogies, key takeaways)
- ✅ A critical review / gap analysis
- ✅ A final polished answer stored in long-term vector memory
- ✅ Ability to query your accumulated study notes in future sessions

---

## 🏗️ Architecture (Concepts from the Document)

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              MULTI-AGENT ORCHESTRATOR (LangGraph)        │
│                                                          │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐       │
│   │ RESEARCHER │───▶│  ANALYST  │───▶│  CRITIC   │       │
│   │  Agent    │    │  Agent    │    │  Agent    │       │
│   │ (RAG +    │    │ (CoT +    │    │(Reflexion)│       │
│   │  Search)  │    │  Struct.) │    │           │       │
│   └───────────┘    └───────────┘    └─────┬─────┘       │
│                                           │              │
│                         ┌─────────────────▼──────────┐  │
│                         │     FINALISER Agent         │  │
│                         │   (Memory Write + Output)   │  │
│                         └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   VECTOR MEMORY      │
                     │  (ChromaDB / FAISS)  │
                     │  Long-term storage   │
                     └─────────────────────┘
```

---

## 🧠 Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **Prompt Engineering** | Zero-shot, Few-shot, CoT, Role/Persona prompts per agent |
| **RAG Pipeline** | Load → Chunk → Embed → Store → Retrieve → Generate |
| **Embeddings** | sentence-transformers for semantic similarity |
| **Vector Database** | ChromaDB for persistent long-term memory |
| **Agent Loop** | Perceive → Think → Act → Observe cycle |
| **Multi-Agent System** | Researcher, Analyst, Critic, Finaliser roles |
| **ReAct Pattern** | Agents reason then act with tools |
| **Reflexion** | Critic agent self-reflects and routes back if needed |
| **Memory** | In-context (short-term) + ChromaDB (long-term episodic) |
| **Tool Use** | Wikipedia search, web fetch, calculator tools |
| **Cognitive Arch.** | MRKL-style modular reasoning |
| **Human-in-the-Loop** | Optional approval gate before finalizing |
| **State Management** | TypedDict state flows through all agents |
| **Conditional Routing** | Critic routes to re-analyse OR finalise |

---

## 🚀 Quick Start

### 1️⃣ Click & Try (no setup)
Jump straight in: **[Open Live App](https://ghanendrasahu-multi-agent-study-assistant-app-ghxuhm.streamlit.app/)** — pick a query and run it.

### 2️⃣ Run Locally

```bash
# Clone the repo
git clone https://github.com/ghanendrasahu/Multi-Agent-Study-Assistant.git
cd Multi-Agent-Study-Assistant

# Install dependencies
pip install -r requirements.txt

# Set your API key
set OPENAI_API_KEY=sk-...           # Windows
export OPENAI_API_KEY="sk-..."      # macOS / Linux

# Launch the web UI
streamlit run app.py
```

Or use the CLI:
```bash
PYTHONPATH=src python src/study_agent/main.py
```

---

## 📁 Project Structure

```
Multi-Agent-Study-Assistant/
├── app.py                 ← Streamlit Cloud entry point
├── pyproject.toml         ← Pip-installable package config
├── Makefile               ← Dev commands (make install, make run-ui)
├── requirements.txt       ← Python dependencies
├── .env                   ← Your API key (gitignored)
├── src/
│   └── study_agent/
│       ├── app.py              # Streamlit web UI
│       ├── main.py             # CLI entry point
│       ├── llm_factory.py      # OpenAI / Ollama / Anthropic provider
│       ├── orchestrator.py     # LangGraph-style graph builder
│       ├── agents/
│       │   ├── state.py        # Shared TypedDict (LangGraph pattern)
│       │   ├── researcher.py   # RAG-powered research agent
│       │   ├── analyst.py      # CoT + structured output agent
│       │   ├── critic.py       # Reflexion-pattern self-critique
│       │   └── finaliser.py    # Memory-write + output formatting
│       ├── tools/
│       │   ├── search.py       # Wikipedia + DuckDuckGo search
│       │   └── calculator.py   # Safe AST-based math evaluator
│       └── memory/
│           └── vector_store.py # ChromaDB long-term memory
│       (plus __init__.py files for each package)
├── data/
│   └── chroma_db/         ← Persistent vector store (gitignored)
├── docs/                  # Additional documentation
└── tests/                 # Test suite
```

---

## 💡 Interview Talking Points

1. **Why multi-agent?** Single agents hit context limits; specialized agents are more reliable
2. **Why RAG?** Grounds answers in retrieved knowledge, reduces hallucination  
3. **Why ChromaDB?** Persistent semantic memory across sessions — the agent "remembers" what it studied
4. **Reflexion pattern** — the Critic agent scores the answer; below threshold → loops back to Analyst
5. **State management** — TypedDict ensures every agent reads/writes the same keys safely
6. **Fallback design** — max_iterations guard prevents infinite critique loops
7. **Tool abstraction** — tools are just Python functions with typed inputs; the LLM decides when to call them

---

## 🐞 Report an Issue

Found a bug or have a suggestion? [Open an issue](https://github.com/ghanendrasahu/Multi-Agent-Study-Assistant/issues) on GitHub.
