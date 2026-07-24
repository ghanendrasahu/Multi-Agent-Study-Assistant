# 🤖 Multi-Agent Study Assistant

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

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
# Option A: OpenAI
export OPENAI_API_KEY="sk-..."

# Option B: Use FREE local mode (Ollama)
# Install Ollama, pull llama3: ollama pull llama3
# Then set: export USE_OLLAMA=true
```

### 3. Run the CLI
```bash
python main.py
```

### 4. Run the Web UI
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
study_agent/
├── main.py              # CLI entry point
├── app.py               # Streamlit web UI
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── state.py         # Shared TypedDict state (LangGraph pattern)
│   ├── researcher.py    # RAG-powered research agent
│   ├── analyst.py       # CoT analysis + structured output agent
│   ├── critic.py        # Reflexion-pattern self-critique agent
│   └── finaliser.py     # Memory-write + output formatting agent
├── tools/
│   ├── __init__.py
│   ├── search.py        # Wikipedia + DuckDuckGo search tools
│   └── calculator.py    # Math tool (demonstrates tool use)
├── memory/
│   ├── __init__.py
│   └── vector_store.py  # ChromaDB long-term memory
└── orchestrator.py      # LangGraph-style graph builder & runner
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
