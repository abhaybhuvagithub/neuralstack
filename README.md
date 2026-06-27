# NeuralStack — RAG + Agent + MCP on the Anthropic API

A compact, readable reference for *agentic* retrieval-augmented generation.
Instead of stuffing retrieved text into the prompt up front, retrieval is
exposed to Claude as a **tool** (`search_knowledge_base`). The model decides
when to search, can search more than once, and can combine search with other
tools (a calculator here, or any MCP server you attach).

```
documents ──chunk──► embed ──► vector store
                                    ▲
question ──► Agent (Claude) ──tool: search_knowledge_base──┘
                  │
                  └──► grounded, source-cited answer
```

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python main.py "How is gratuity calculated in India?"
```

The ingestion and retrieval half runs with **no API key** — `main.py` prints
the passages it would retrieve before any model call, so you can judge
grounding quality for free.

## Layout

| File | Role |
|------|------|
| `embeddings.py` | dependency-free feature-hashing embedder (swap for Voyage AI for real semantics) |
| `vectorstore.py` | in-memory cosine top-k store (swap for Chroma/Pinecone at scale) |
| `ingest.py` | overlapping-window chunking + document loading |
| `tools.py` | tool JSON schemas + safe calculator |
| `mcp_client.py` | builds the `mcp_servers` payload to plug in remote MCP tools |
| `agent.py` | the tool-use loop that orchestrates everything |
| `main.py` | wiring + a runnable demo over the sample `data/` corpus |

## Adding MCP servers

MCP lets Claude call tools hosted elsewhere (issue trackers, databases,
internal services) without you proxying each call. Attach them in `main.py`:

```python
from neuralstack.mcp_client import build_mcp_servers
servers = build_mcp_servers([
    {"url": "https://mcp.example.com/sse", "name": "issues", "token": "..."},
])
agent = Agent(store, mcp_servers=servers)
```

When `mcp_servers` is set, the agent automatically switches to the beta
Messages endpoint and sends the required MCP beta header.

## What to harden for production

- Real embeddings (Voyage AI) and an ANN index or managed vector DB.
- Token-aware chunking on semantic boundaries.
- Reranking the top-k before it reaches the model.
- Streaming responses and per-tool timeouts/retries.

## AI Periodic Table
https://htmlpreview.github.io/?https://github.com/abhaybhuvagithub/neuralstack/blob/main/ai-periodic-table_with_Quiz_OOB.html

