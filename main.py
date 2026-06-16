"""main.py — wire the pieces together and ask a question.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py "How is gratuity calculated for Indian employees?"

The ingestion + retrieval half runs with no API key (it's all local), so
you can inspect what would be retrieved before spending any tokens.
"""
from __future__ import annotations

import os
import sys

from neuralstack import HashingEmbedder, VectorStore, load_documents, Agent
# from neuralstack.mcp_client import build_mcp_servers  # see README for MCP


def build_store(data_dir: str) -> VectorStore:
    store = VectorStore(embedder=HashingEmbedder(dim=512))
    chunks = load_documents(data_dir)
    store.add_many(chunks)
    print(f"Indexed {len(store)} chunks from {data_dir}")
    return store


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    store = build_store(os.path.join(here, "data"))

    question = " ".join(sys.argv[1:]) or "What does the knowledge base cover?"

    # Preview retrieval (no API call) so you can see grounding quality.
    print(f"\nQuery: {question}\nTop retrieved passages:")
    for rec, score in store.search(question, k=3):
        print(f"  [{score:.2f}] {rec.metadata['source']}: {rec.text[:80]}...")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n(Set ANTHROPIC_API_KEY to run the full agent loop.)")
        return

    # To add MCP servers:
    #   servers = build_mcp_servers([{"url": "...", "name": "...", "token": "..."}])
    #   agent = Agent(store, mcp_servers=servers)
    agent = Agent(store)
    print("\nAgent answer:\n" + agent.ask(question))


if __name__ == "__main__":
    main()
