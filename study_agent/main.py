"""
main.py
───────
CLI entry point for the Multi-Agent Study Assistant.

Usage:
    python main.py
    python main.py --query "Explain transformer self-attention"
    python main.py --topic "RAG" --max-iter 2
    python main.py --list-memory
"""

from __future__ import annotations

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present


def main():
    parser = argparse.ArgumentParser(
        description="🤖 Multi-Agent Study Assistant — GenAI + Agentic AI Demo"
    )
    parser.add_argument("--query", "-q", type=str, help="Study query/question")
    parser.add_argument("--topic", "-t", type=str, help="Topic label for memory")
    parser.add_argument("--max-iter", "-m", type=int, default=2,
                        help="Max analyst-critic iterations (default: 2)")
    parser.add_argument("--list-memory", action="store_true",
                        help="List all topics stored in memory")
    parser.add_argument("--show-graph", action="store_true",
                        help="Show the agent graph topology")
    parser.add_argument("--provider", type=str,
                        choices=["openai", "anthropic", "ollama"],
                        help="LLM provider (auto-detects from env if not set)")
    args = parser.parse_args()

    # ── Try rich for pretty output, fall back to print ────────────────────
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich import print as rprint
        console = Console()
        USE_RICH = True
    except ImportError:
        USE_RICH = False
        console = None

    def header(text: str):
        if USE_RICH:
            console.print(Panel(f"[bold cyan]{text}[/bold cyan]", expand=False))
        else:
            print(f"\n{'='*60}\n{text}\n{'='*60}")

    def info(text: str):
        if USE_RICH:
            console.print(f"[dim]{text}[/dim]")
        else:
            print(text)

    def success(text: str):
        if USE_RICH:
            console.print(f"[green]✅ {text}[/green]")
        else:
            print(f"✅ {text}")

    def show_markdown(text: str):
        if USE_RICH:
            console.print(Markdown(text))
        else:
            print(text)

    # ── Banner ─────────────────────────────────────────────────────────────
    header("🤖 Multi-Agent Study Assistant")
    info("Demonstrating: GenAI + Agentic AI + RAG + Multi-Agent Orchestration")

    # ── Initialise components ──────────────────────────────────────────────
    from llm_factory import get_llm
    from memory.vector_store import VectorMemory
    from orchestrator import build_graph

    try:
        llm = get_llm(provider=args.provider)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    memory = VectorMemory()
    graph = build_graph(llm=llm, memory=memory)

    # ── Show graph topology ────────────────────────────────────────────────
    if args.show_graph:
        print(graph.get_graph_description())
        return

    # ── List memory ────────────────────────────────────────────────────────
    if args.list_memory:
        topics = memory.list_topics()
        count = memory.count()
        header(f"📚 Long-Term Memory ({count} notes stored)")
        if topics:
            for t in sorted(topics):
                print(f"  • {t}")
        else:
            print("  No notes stored yet. Run a query to start building memory.")
        return

    # ── Get query ──────────────────────────────────────────────────────────
    query = args.query
    if not query:
        print("\n" + "─" * 60)
        print("💡 Example queries:")
        print("  • Explain transformer self-attention mechanism")
        print("  • What is RAG and how does it reduce hallucination?")
        print("  • Explain the ReAct agent pattern with an example")
        print("  • What is cosine similarity and why use it for embeddings?")
        print("  • Explain LangGraph StateGraph with nodes and edges")
        print("─" * 60 + "\n")
        query = input("Enter your study query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(0)

    topic = args.topic or query[:50]

    # ── Run the agent graph ────────────────────────────────────────────────
    header(f"📖 Studying: {query}")

    agent_steps = []

    def progress_callback(agent: str, status: str):
        agent_steps.append((agent, status))
        emoji = {
            "researcher": "🔍",
            "analyst": "✍️",
            "critic": "🔎",
            "finaliser": "✨"
        }.get(agent.lower(), "⚙️")
        if USE_RICH:
            console.print(f"  {emoji} [bold]{agent.upper()}[/bold]: {status}")
        else:
            print(f"  {emoji} {agent.upper()}: {status}")

    try:
        result = graph.invoke(
            query=query,
            topic=topic,
            max_iterations=args.max_iter,
            progress_callback=progress_callback,
        )
    except Exception as e:
        print(f"\n❌ Agent pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Display final answer ───────────────────────────────────────────────
    header("📝 Final Study Note")
    show_markdown(result["final_answer"])

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    info(f"Quality score: {result['critique_score']:.2f}/1.00")
    info(f"Iterations: {result['iteration_count']}")
    info(f"Memory notes stored: {memory.count()}")
    info(f"Session ID: {result['session_id']}")
    success("Study note saved to long-term memory!")

    # ── Option to query memory ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    another = input("\n📚 Ask another question? (or press Enter to exit): ").strip()
    if another:
        # Recursive call for another query (demonstrates memory retrieval)
        import subprocess
        subprocess.run([sys.executable, __file__, "--query", another,
                        "--provider", args.provider or "openai"])


if __name__ == "__main__":
    main()
