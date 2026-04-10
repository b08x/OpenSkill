"""
openskill CLI — The Swiss Army Knife for Geometric Skill Memory
==============================================================
Commands:
    create   - MemCollab: Create a new skill from a task.
    retrieve - S-Path-RAG: Intelligent search (text or vector injection).
    evolve   - Trace2Skill: Improve skill based on trajectories.
    list     - List local library.
    graph    - Inspect the topology of the skill brain.
    embed    - Force TurboQuant re-quantization of a skill.
    serve    - Start the MCP/FastAPI server for Cursor IDE.
"""

from __future__ import annotations

import asyncio
import sys
import json
import os
from pathlib import Path
from typing import Optional

import click
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from openskill import OpenSkillClient, LocalDiskStore
from openskill.llm.openrouter import OpenRouterProvider
from openskill.llm.ollama import OllamaProvider
from openskill.core.vector import unpack_qvector

# Logger and Console for rich UI in the terminal
log = structlog.get_logger()
console = Console()

def get_client(skill_dir: str, api_key: Optional[str], local: bool, model_id: Optional[str] = None) -> OpenSkillClient:
    """
    Client factory.
    If 'local' is True, attempts to load the local vector injection engine.
    """
    store = LocalDiskStore(skill_dir)

    if local:
        # If local, we use the provider that supports vector injection (LocalSkillInjectedLLM)
        from openskill.injection.local_llm import LocalSkillInjectedLLM
        # Default to the Qwen3.5 that the user downloaded
        mid = model_id or "Qwen/Qwen3.5-2B"
        console.print(f"[bold yellow]Loading Local Engine ([/bold yellow]{mid}[bold yellow])...[/bold yellow]")
        llm = LocalSkillInjectedLLM(model_id=mid)
    elif api_key:
        llm = OpenRouterProvider(api_key=api_key)
    else:
        key = os.getenv("OPENROUTER_API_KEY", "")
        llm = OpenRouterProvider(api_key=key) if key else None

    return OpenSkillClient(store=store, llm=llm)

@click.group()
@click.version_option()
def cli():
    """OpenSkill — Unified lifecycle for agent skills."""
    pass

# ── COMMAND: CREATE (MemCollab) ──────────────────────────────────────────────

@cli.command()
@click.argument("task")
@click.option("--skill-dir", default="./skills_output", help="Skills directory")
@click.option("--api-key", help="OpenRouter API Key")
@click.option("--weak", default="openai/gpt-4o-mini", help="Weak Agent Model")
@click.option("--strong", default="anthropic/claude-3-5-sonnet", help="Strong Agent Model")
def create(task: str, skill_dir: str, api_key: str, weak: str, strong: str):
    """Create a new skill using MemCollab contrastive analysis."""
    client = get_client(skill_dir, api_key, local=False)

    async def _run():
        with console.status("[bold green]Distilling skill via MemCollab..."):
            meta = await client.craft(task=task, weak_model=weak, strong_model=strong)

        console.print(Panel(
            f"[bold str]ID:[/bold str] {meta.id}\n"
            f"[bold str]Title:[/bold str] {meta.title}\n"
            f"[bold str]Category:[/bold str] {meta.category}/{meta.subcategory}",
            title="Skill Created Successfully", border_style="green"
        ))

    asyncio.run(_run())

# ── COMMAND: RETRIEVE (S-Path-RAG + TurboQuant) ──────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--skill-dir", default="./skills_output")
@click.option("--local", is_flag=True, help="Use Local Vector Injection")
@click.option("--use-gnn", is_flag=True, help="Force search using GNN-enriched vectors")
# --- FIX: Added 'cross_attention' and 'prefix' to mode choices ---
@click.option("--mode", default="auto",
              type=click.Choice(["injection", "prefix", "cross_attention", "verbalization", "auto"]),
              help="Generation mode")
@click.option("--model-id", help="Model ID for local execution")
@click.option("--top-k", default=3)
def retrieve(query: str, skill_dir: str, local: bool, use_gnn: bool, mode: str, model_id: str, top_k: int):
    """Search skills and generate response (TurboQuant + S-Path-RAG)."""

    client = get_client(skill_dir, None, local=local, model_id=model_id)

    async def _run():
        with console.status("[bold blue]Navigating Skill Graph (Neural-Socratic)..."):
            # 1. Retrieve Guidance
            guidance = await client.retriever.retrieve(query, top_k=top_k, use_gnn=use_gnn)

        # Show the graph trace
        console.print(f"\n[bold]Socratic Trace:[/bold]\n[dim]{guidance.reasoning_trace}[/dim]\n")

        # If in local mode, generate with the retrieved guidance
        if local and hasattr(client.llm, 'generate_with_guidance'):
            with console.status(f"[bold magenta]Generating answer (mode={mode})..."):
                resp = await client.llm.generate_with_guidance(query, guidance, mode=mode)

            # UI adjustments for panel display
            conf = guidance.confidence if hasattr(guidance, 'confidence') else 0.0
            panel_title = f"Generated Answer | mode={mode} | conf={conf:.2f}"
            console.print(Panel(resp.content, title=panel_title, border_style="magenta"))
        else:
            # Just show the found skills (Mode without local generation)
            for i, content in enumerate(guidance.skill_contents):
                console.print(f"[bold cyan]Skill {i + 1}:[/bold cyan]")
                console.print(Markdown(content))
                console.print("-" * 40)

    asyncio.run(_run())


@cli.command()
@click.option("--skill-dir", default="./skills_output")
@click.option("--api-key", help="OpenRouter API key to generate data")
@click.option("--epochs", default=50, help="Number of training epochs")
def train_bootstrap(skill_dir: str, api_key: str, epochs: int):
    """
    Phase 2: Uses the Strong LLM to generate data and trains the local Path Scorer (384d).
    """
    from openskill.core.trainer import train_path_scorer
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # We use OpenRouter only to generate the TEXT for the queries
    client = get_client(skill_dir, api_key, local=False)

    async def _run():
        console.print("[bold blue]Starting Bootstrap Training (Phase 2)...[/bold blue]")

        # Load the 384d embedder explicitly to align the data
        console.print("Loading local embedder for training alignment (384d)...")
        local_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        all_metas = await client.store.list_skills()
        if len(all_metas) < 2:
            console.print("[red]Error: You need at least 2 skills for contrastive training.[/red]")
            return

        train_data = []

        for meta in all_metas:
            console.print(f" Generating samples for: [cyan]{meta.title}[/cyan]")

            # 1. Generate text via Strong LLM
            prompt = f"""Generate 10 diverse user queries that would be solved by this technical skill. 
            Format: Just the queries, one per line.
            SKILL TITLE: {meta.title}
            SKILL CONTENT: {meta.task}"""

            from openskill.llm.base import LLMMessage
            resp = await client.llm.generate([LLMMessage(role="user", content=prompt)])
            queries = [q.strip() for q in resp.content.split("\n") if q.strip() and len(q) > 10]

            # 2. Get local skill vector (384d)
            skill_vec = None
            vectors_dict = getattr(meta, 'vectors', {})
            for p in vectors_dict.values():
                if getattr(p, 'dimension', 0) == 384:
                    skill_vec = np.array(p.embedding)
                    break

            if skill_vec is None:
                console.print(f"[yellow]Warning: Skill {meta.id} does not have a 384d profile. Skipping...[/yellow]")
                continue

            # 3. Create Positive and Negative pairs
            for q_text in queries:
                # QUERY EMBEDDING (Forced to 384d to match local skill)
                q_vec = local_embedder.encode(q_text, normalize_embeddings=True)

                # POSITIVE EXAMPLE
                train_data.append((q_vec, skill_vec, True))

                # NEGATIVE EXAMPLE (Pick a random skill that is not this one)
                other_metas = [m for m in all_metas if m.id != meta.id]
                if other_metas:
                    other_meta = np.random.choice(other_metas)
                    other_vec = None
                    for p in other_meta.vectors.values():
                        if getattr(p, 'dimension', 0) == 384:
                            other_vec = np.array(p.embedding)
                            break

                    if other_vec is not None:
                        train_data.append((q_vec, other_vec, False))

        if not train_data:
            console.print("[red]Error: No training data generated.[/red]")
            return

        # 4. TRIGGER TRAINING
        save_path = str(client.store.workspace_path / "path_scorer.safetensors")
        console.print(f"[bold yellow]Training neural scorer on {len(train_data)} samples...[/bold yellow]")

        await train_path_scorer(
            train_data=train_data,
            embed_dim=384,  # Now guaranteed!
            save_path=save_path,
            epochs=epochs
        )

        console.print(f"[bold green]✓ Phase 2 Complete! Scorer saved to {save_path}[/bold green]")

    asyncio.run(_run())


@cli.command()
@click.option("--skill-dir", default="./skills_output")
@click.option("--use-gnn", is_flag=True, help="Enables neural GNN refinement")
def build_graph(skill_dir: str, use_gnn: bool):
    """Scan all skills and rebuild the graph (with GNN option)."""
    from openskill.core.graph import register_skill_in_graph
    client = get_client(skill_dir, None, local=False)

    async def _run():
        console.print(f"[bold blue]Rebuilding Skill Graph {'(Neural GNN Mode)' if use_gnn else ''}...")

        all_metas = await client.store.list_skills()
        metas_dict = {m.id: m.to_dict() for m in all_metas}

        # Clear current graph to avoid duplicates
        from openskill.storage.base import SkillGraphData
        await client.store.update_graph(SkillGraphData())

        for meta in all_metas:
            # Pass use_gnn to the registration function
            await register_skill_in_graph(
                meta.id, meta, metas_dict, client.store, use_gnn=use_gnn
            )
            console.print(f" Registered: {meta.title}")

        console.print("[bold green]✓ Graph and Encodings Rebuilt Successfully.")

    asyncio.run(_run())

# ---- CONVERT

@cli.command()
@click.argument("skill-id")
@click.option("--skill-dir", default="./skills_output", help="Skills directory")
@click.option("--local", is_flag=True, help="Convert to 384d (Local)")
@click.option("--api-key", help="Convert to 1536d (Remote)")
def convert(skill_id: str, local: bool, api_key: str, skill_dir: str):
    """Generate a new geometric 'View' for an existing skill."""
    # Now passing skill_dir from options (or default)
    sid = skill_id.strip()
    client = get_client(skill_dir, api_key, local=local)

    async def _run():
        # DEBUG: Print search path
        # console.print(f"[dim]Checking path: {client.store.skills_dir}[/dim]")

        meta = await client.store.get_skill_meta(skill_id)
        content = await client.store.get_skill_md(skill_id)

        if not meta or not content:
            console.print(f"[red]Error: Skill {skill_id} not found in {client.store.skills_dir}[/red]")
            return

        target = "LOCAL (384d)" if local else "REMOTE (1536d)"
        with console.status(f"[bold yellow]Converting skill {skill_id} to {target}..."):
            await client._embed_and_register(skill_id, content, meta)

        console.print(f"[green]✓ Skill {skill_id} converted to {target}.[/green]")

    asyncio.run(_run())

# ── COMMAND: EVOLVE (Trace2Skill) ────────────────────────────────────────────

@cli.command()
@click.argument("skill-id")
@click.option("--skill-dir", default="./skills_output")
@click.option("--tasks", help="Comma-separated tasks for testing")
def evolve(skill_id: str, skill_dir: str, tasks: str):
    """Evolve a skill based on error evidence (Trace2Skill)."""
    client = get_client(skill_dir, None, local=False)

    async def _run():
        task_list = [t.strip() for t in tasks.split(",")] if tasks else []
        with console.status(f"[bold yellow]Evolving skill {skill_id} in parallel fleet..."):
            result = await client.evolve(skill_id=skill_id, tasks=task_list)

        console.print(f"[bold green]Success![/bold green] Applied {result['patch_count']} patches.")
        console.print(f"Fleet Success Rate: {result['success_rate']*100:.1f}%")

    asyncio.run(_run())

# ── COMMAND: LIST ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--skill-dir", default="./skills_output")
def list(skill_dir: str):
    """List all skills and geometric memory status."""
    client = get_client(skill_dir, None, local=False)

    async def _run():
        metas = await client.list_skills()
        table = Table(title="OpenSkill Library")
        table.add_column("ID", style="dim")
        table.add_column("Title", style="cyan")
        table.add_column("Category")
        table.add_column("TurboQuant", justify="center")
        table.add_column("Evolutions", justify="center")

        for m in metas:
            status = "[green]✓[/green]" if m.qvector else "[red]✗[/red]"
            table.add_row(m.id, m.title, f"{m.category}/{m.subcategory}", status, str(m.evolution_count))

        console.print(table)

    asyncio.run(_run())

# ── COMMAND: GRAPH ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--skill-dir", default="./skills_output")
def graph(skill_dir: str):
    """Visualize knowledge graph topology."""
    store = LocalDiskStore(skill_dir)
    g = store.get_graph()

    console.print(f"[bold]Nodes:[/bold] {len(g.nodes)} | [bold]Edges:[/bold] {len(g.edges)}")

    table = Table(show_header=False, box=None)
    for e in g.edges:
        table.add_row(f"[cyan]{e['from']}[/cyan]", f"──([italic]{e['type']}[/italic])──>", f"[cyan]{e['to']}[/cyan]")

    console.print(table)

# ── COMMAND: EMBED (TurboQuant Fix) ──────────────────────────────────────────

@cli.command()
@click.argument("skill-id")
@click.option("--skill-dir", default="./skills_output")
@click.option("--api-key", help="OpenRouter API key")
@click.option("--local", is_flag=True, help="Use local engine")
def embed(skill_id: str, skill_dir: str, api_key: str, local: bool):
    """Recalculate TurboQuant vector for a specific skill."""
    # Factory now correctly receives parameters
    client = get_client(skill_dir, api_key, local=local)

    async def _run():
        if client.llm is None:
            console.print("[red]Error: LLM provider not configured. Use --api-key or --local.[/red]")
            return

        with console.status("[bold blue]Recalculating TurboQuant geometry..."):
            meta = await client.store.get_skill_meta(skill_id)
            content = await client.store.get_skill_md(skill_id)

            if not meta or not content:
                console.print(f"[red]Skill {skill_id} not found on disk.[/red]")
                return

            # Call adjusted helper in client.py
            await client._embed_and_register(skill_id, content, meta)

        console.print(f"[green]✓ Skill {skill_id} quantized and registered in graph.[/green]")

    asyncio.run(_run())

# ── COMMAND: SERVE (FastAPI/MCP) ──────────────────────────────────────────────

@cli.command()
@click.option("--port", default=8000)
@click.option("--skill-dir", default="./skills_output")
def serve(port: int, skill_dir: str):
    """Start the API server and MCP endpoint for Cursor IDE."""
    import uvicorn
    # Late import to keep CLI light
    from openskill.mcp.server import main as run_mcp

    console.print(f"[bold green]Starting OpenSkill Server on port {port}...[/bold green]")
    console.print(f"MCP Endpoint active for Cursor/Windsurf.")

    # Here you can choose to run the MCP server or the FastAPI app
    # For Cursor, MCP via stdio is the default.
    asyncio.run(run_mcp())

def main():
    cli()

if __name__ == "__main__":
    main()
