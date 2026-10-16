"""A local stdio MCP server that serves fixed tool schemas and canned answers.

The gate needs an MCP server whose tools cost the same prompt tokens as the
operator's real one, without the real one's network calls. This serves the
schemas in a JSON file (a list of ``{"name", "description", "inputSchema"}``)
verbatim, and answers every call with one line naming the tool and its
arguments, so a transcript shows the call landed.

Usage: python stub_mcp.py <tools.json>
"""

from __future__ import annotations

import asyncio
import json
import sys

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

with open(sys.argv[1], encoding="utf-8") as _fh:
    TOOLS = json.load(_fh)

server = Server("local-gate-stub")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=t["name"], description=t.get("description", ""), inputSchema=t["inputSchema"])
        for t in TOOLS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    text = f"[local-gate stub] {name} called with {json.dumps(arguments, sort_keys=True)}"
    return [types.TextContent(type="text", text=text)]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
