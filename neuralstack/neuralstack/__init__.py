"""NeuralStack — a compact RAG + Agent + MCP reference built on the Anthropic API.

Pipeline:  documents -> chunk -> embed -> vector store
           question  -> agent loop (Claude + tools) -> grounded answer

The agent is given a `search_knowledge_base` tool backed by the vector
store, so retrieval happens *as a tool call the model decides to make*
(agentic RAG) rather than being bolted on before the prompt.
"""

from .embeddings import HashingEmbedder
from .vectorstore import VectorStore
from .ingest import chunk_text, load_documents
from .agent import Agent

__all__ = ["HashingEmbedder", "VectorStore", "chunk_text", "load_documents", "Agent"]
