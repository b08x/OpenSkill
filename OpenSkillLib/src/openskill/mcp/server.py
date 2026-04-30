"""
MCP Server — Model Context Protocol for IDEs
==============================================
Exposes OpenSkill as an MCP tool for:
  - Cursor IDE
  - Windsurf (Codeium)
  - Claude Desktop (Anthropic)
  - Any MCP-compatible client

Usage in Cursor:
  1. Add to ~/.cursor/mcp.json:
     {
       "mcpServers": {
         "openskill": {
           "command": "python",
           "args": ["-m", "openskill.mcp.server"]
         }
       }
     }

  2. In Cursor, ask: "Use the Raft consensus skill to help me
     handle a network partition"
"""

from __future__ import annotations

import asyncio
import json
import structlog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# MCP SDK — uses official Anthropic spec
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    from mcp.server.notification import NotificationOptions
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from openskill import OpenSkillClient
from openskill.storage.local import LocalDiskStore
from openskill.utils.config import get_openrouter_key

log = structlog.get_logger()


@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    skill_dir: str = "./skills_output"
    default_weak_model: str = "openai/gpt-4o-mini"
    default_strong_model: str = "anthropic/claude-3-5-sonnet"
    api_key: Optional[str] = field(default_factory=get_openrouter_key)


@dataclass
class MCPToolContext:
    """Context shared between MCP tools."""
    client: OpenSkillClient
    config: MCPServerConfig


# ── Available MCP Tools ──────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "openskill_craft",
        "description": (
            "Creates a new reasoning skill from a problem. "
            "Uses contrastive analysis between a weak and a strong model to "
            "extract invariants, violation patterns, and normative constraints. "
            "Returns a structured skill that can be used to guide future generations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the problem or task",
                    "example": "Implement Raft consensus over a high-latency network"
                },
                "weak_model": {
                    "type": "string",
                    "description": "Weaker model (fallback)",
                    "default": "openai/gpt-4o-mini"
                },
                "strong_model": {
                    "type": "string",
                    "description": "Stronger model (target)",
                    "default": "anthropic/claude-3-5-sonnet"
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "openskill_retrieve",
        "description": (
            "Searches for relevant skills to solve a problem using "
            "TurboQuant vector search + S-Path-RAG semantic graph. "
            "Returns normative constraints and reasoning patterns that "
            "should be applied when solving the problem."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question or problem",
                    "example": "How to handle leader election failure in a distributed system?"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of skills to return",
                    "default": 3,
                },
                "format": {
                    "type": "string",
                    "enum": ["constraints", "full", "summary"],
                    "description": "Return format",
                    "default": "constraints",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "openskill_evolve",
        "description": (
            "Evolves an existing skill from execution trajectories. "
            "Uses Trace2Skill with a fleet of sub-agents to propose patches "
            "that are consolidated hierarchically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "ID of the skill to evolve",
                },
                "trajectories": {
                    "type": "array",
                    "description": "List of trajectories [{task, trajectory, success}]",
                    "items": {"type": "object"},
                },
                "tasks": {
                    "type": "array",
                    "description": "Or: list of tasks to generate trajectories automatically",
                    "items": {"type": "string"},
                },
            },
            "required": ["skill_id"],
        },
    },
    {
        "name": "openskill_list",
        "description": "Lists all available skills in the local repository.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "openskill_graph",
        "description": "Returns the relationship graph between skills.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Tool Call Handler ───────────────────────────────────────────────────

async def handle_tool_call(
    ctx: MCPToolContext,
    tool_name: str,
    arguments: dict,
) -> TextContent:
    """MCP tool call dispatch."""

    try:
        if tool_name == "openskill_craft":
            meta = await ctx.client.craft(
                task=arguments["task"],
                weak_model=arguments.get("weak_model"),
                strong_model=arguments.get("strong_model"),
            )
            return TextContent(
                type="text",
                text=json.dumps({
                    "status": "created",
                    "skill_id": meta.id,
                    "title": meta.title,
                    "category": f"{meta.category}/{meta.subcategory}",
                    "message": (
                        f"Skill '{meta.title}' created successfully. "
                        f"ID: {meta.id}. Use openskill_retrieve to search for guidance."
                    ),
                }, indent=2),
            )

        elif tool_name == "openskill_retrieve":
            result = await ctx.client.retrieve(
                query=arguments["query"],
                top_k=arguments.get("top_k", 3),
            )

            fmt = arguments.get("format", "constraints")
            skills = result.get("skills", [])

            if fmt == "summary":
                output = "\n".join(
                    f"• [{s.get('title', '?')}] — {s.get('domain', '')}"
                    for s in skills
                )
            elif fmt == "full":
                output = json.dumps(result, indent=2, default=str)
            else:  # constraints
                lines = []
                for i, s in enumerate(skills, 1):
                    md = s.get("content", "")
                    # Extracts only constraints from markdown
                    lines.append(f"## Skill {i}: {s.get('title', '?')}")
                    if "## Normative Constraints" in md:
                        start = md.index("## Normative Constraints")
                        end = md.index("\n##", start + 1) if "\n##" in md[start+1:] else len(md)
                        lines.append(md[start:end])
                    lines.append("")
                output = "\n".join(lines)

            return TextContent(
                type="text",
                text=output or "No skills found for this query.",
            )

        elif tool_name == "openskill_evolve":
            result = await ctx.client.evolve(
                skill_id=arguments["skill_id"],
                trajectories=arguments.get("trajectories"),
                tasks=arguments.get("tasks"),
            )
            return TextContent(
                type="text",
                text=json.dumps({
                    "status": "evolved",
                    "patch_count": result["patch_count"],
                    "fleet_size": result["fleet_size"],
                    "success_rate": f"{result['success_rate']:.1%}",
                }, indent=2),
            )

        elif tool_name == "openskill_list":
            metas = await ctx.client.list_skills()
            return TextContent(
                type="text",
                text=json.dumps([
                    {
                        "id": m.id,
                        "title": m.title,
                        "category": f"{m.category}/{m.subcategory}",
                        "domain": m.domain,
                        "evolution_count": m.evolution_count,
                        "created_at": m.created_at,
                    }
                    for m in metas
                ], indent=2, default=str),
            )

        elif tool_name == "openskill_graph":
            graph = ctx.client.store.get_graph()
            return TextContent(
                type="text",
                text=json.dumps(graph.to_dict(), indent=2, default=str),
            )

        else:
            return TextContent(
                type="text",
                text=f"Unknown tool: {tool_name}",
            )

    except Exception as e:
        log.error("mcp_tool.error", tool=tool_name, error=str(e))
        return TextContent(
            type="text",
            text=f"Error executing {tool_name}: {e}",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(config: MCPServerConfig | None = None) -> None:
    """Runs the MCP server via stdio."""

    if not MCP_AVAILABLE:
        print(
            "ERROR: mcp package not installed.\n"
            "Install with: pip install openskill[mcp]",
            file=sys.stderr,
        )
        return

    import sys

    cfg = config or MCPServerConfig()

    # Initializes OpenSkill client
    from openskill.llm.openrouter import OpenRouterProvider
    llm = OpenRouterProvider(api_key=cfg.api_key or "") if cfg.api_key else None
    store = LocalDiskStore(cfg.skill_dir)
    client = OpenSkillClient(store=store, llm=llm)

    ctx = MCPToolContext(client=client, config=cfg)
    server = Server("openskill")

    # ── Register capabilities ────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["input_schema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, Any],
    ) -> list[TextContent]:
        result = await handle_tool_call(ctx, name, arguments)
        return [result]

    # ── Run stdio server ──────────────────────────────────────────────────
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(
                notification_options=NotificationOptions(),
            ),
        )


def run() -> None:
    """Entry point for: python -m openskill.mcp.server"""
    import sys
    asyncio.run(main())


if __name__ == "__main__":
    run()
