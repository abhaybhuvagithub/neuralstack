"""mcp_client.py — attach remote MCP servers to the agent.

The Model Context Protocol (MCP) lets Claude call tools hosted by external
servers (GitHub, a database gateway, an internal service, ...). The
Anthropic Messages API can connect to these directly via the `mcp_servers`
parameter, so you don't have to proxy every tool call yourself — Claude
talks to the MCP server and you just receive the results.

This module builds the `mcp_servers` payload. It's optional: if you pass no
servers, the agent runs with only its local tools.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Beta header required for the MCP connector on the Messages API.
MCP_BETA_FLAG = "mcp-client-2025-04-04"


def mcp_server(url: str, name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Describe one MCP server. `token` is forwarded as a bearer credential
    when the server requires authentication."""
    server: Dict[str, Any] = {"type": "url", "url": url, "name": name}
    if token:
        server["authorization_token"] = token
    return server


def build_mcp_servers(configs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Turn a list of {url, name, token?} dicts into the API payload.

    Example:
        build_mcp_servers([
            {"url": "https://mcp.example.com/sse", "name": "issues",
             "token": "..."},
        ])
    """
    return [mcp_server(c["url"], c["name"], c.get("token")) for c in configs]
