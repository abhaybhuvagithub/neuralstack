"""agent.py — the agentic loop.

This drives a conversation with Claude where the model decides when to call
tools. The flow per turn:

    1. Send the running message history + tool menu to the model.
    2. If the model returns `stop_reason == "tool_use"`, run each requested
       tool, append the results, and loop.
    3. Otherwise we have the final answer — return it.

`search_knowledge_base` is bound to the live vector store here, so retrieval
is just another tool the model can choose to use (agentic RAG).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .tools import LOCAL_TOOL_FNS, TOOL_SCHEMAS
from .vectorstore import VectorStore
from .mcp_client import MCP_BETA_FLAG

SYSTEM_PROMPT = (
    "You are NeuralStack, a retrieval-augmented assistant. "
    "Prefer answering from the knowledge base: call search_knowledge_base "
    "before relying on prior knowledge. Cite the source filename for any "
    "fact you draw from a retrieved passage. If the documents do not contain "
    "the answer, say so plainly rather than guessing."
)

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 6  # hard stop so a tool loop can never run forever


class Agent:
    def __init__(
        self,
        store: VectorStore,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        model: str = MODEL,
    ):
        # Imported lazily so the rest of the package works without the SDK.
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.store = store
        self.model = model
        self.mcp_servers = mcp_servers or []

    # --- tool execution --------------------------------------------------

    def _run_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a locally-handled tool and return a string result."""
        if name == "search_knowledge_base":
            hits = self.store.search(args["query"], k=int(args.get("k", 3)))
            if not hits:
                return "No matching passages found."
            return "\n\n".join(
                f"[source: {rec.metadata.get('source')} #{rec.metadata.get('chunk')} "
                f"score={score:.2f}]\n{rec.text}"
                for rec, score in hits
            )
        if name in LOCAL_TOOL_FNS:
            return LOCAL_TOOL_FNS[name](**args)
        return f"error: unknown tool '{name}'"

    # --- the loop --------------------------------------------------------

    def ask(self, question: str, verbose: bool = True) -> str:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]

        for _ in range(MAX_ITERATIONS):
            # The MCP connector lives on the beta endpoint; without MCP
            # servers we use the standard endpoint.
            create = self.client.messages.create
            kwargs: Dict[str, Any] = dict(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            if self.mcp_servers:
                create = self.client.beta.messages.create
                kwargs["mcp_servers"] = self.mcp_servers
                kwargs["betas"] = [MCP_BETA_FLAG]

            response = create(**kwargs)

            # Record the assistant turn verbatim (text + tool_use blocks).
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Final answer: concatenate any text blocks.
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

            # Run every tool the model asked for, in order.
            tool_results: List[Dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    print(f"  -> tool: {block.name}({block.input})")
                result = self._run_tool(block.name, dict(block.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return "Stopped: reached the tool-call iteration limit."
