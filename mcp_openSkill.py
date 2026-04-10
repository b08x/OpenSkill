from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json

# O FastMCP com SSE permite conexões remotas via URL
# host="0.0.0.0" é necessário para rodar dentro do Docker
mcp = FastMCP("SkillCrafter_Remote", host="0.0.0.0", port=8001)
SKILLS_DIR = Path("skills_output")

@mcp.tool()
async def search_skills(query: str, category: str = None) -> str:
    """Busca skills por query ou categoria. Use para encontrar a melhor skill para seu problema."""
    results = []
    for meta_file in SKILLS_DIR.glob("*.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        # Lógica simples de filtro
        if query.lower() in meta['task'].lower() or (category and meta.get('category') == category):
            results.append(f"ID: {meta['id']} - {meta['title']} ({meta['category']})")

    return "\n".join(results) if results else "Nenhuma skill encontrada."

@mcp.tool()
async def get_skill_details(skill_id: str) -> str:
    """Retorna o conteúdo da Skill.md pelo ID para o agente ler as regras."""
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists(): return "Skill não encontrada."

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    md_path = SKILLS_DIR / meta["filename"]
    return md_path.read_text(encoding="utf-8")

if __name__ == "__main__":
    # Roda em um servidor web na porta 8001 para evitar conflito com a UI (8000)
    mcp.run(transport="sse")