from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json

# FastMCP with SSE allows remote connections via URL
# host="0.0.0.0" is necessary to run inside Docker
mcp = FastMCP("SkillCrafter_Remote", host="0.0.0.0", port=8001)
SKILLS_DIR = Path("skills_output")

@mcp.tool()
async def search_skills(query: str, category: str = None) -> str:
    """Search skills by query or category. Use to find the best skill for your problem."""
    results = []
    for meta_file in SKILLS_DIR.glob("*.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        # Simple filter logic
        if query.lower() in meta['task'].lower() or (category and meta.get('category') == category):
            results.append(f"ID: {meta['id']} - {meta['title']} ({meta['category']})")

    return "\n".join(results) if results else "No skills found."

@mcp.tool()
async def get_skill_details(skill_id: str) -> str:
    """Returns the Skill.md content by ID for the agent to read the rules."""
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists(): return "Skill not found."

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    md_path = SKILLS_DIR / meta["filename"]
    return md_path.read_text(encoding="utf-8")

if __name__ == "__main__":
    # Runs on a web server at port 8001 to avoid conflict with the UI (8000)
    mcp.run(transport="sse")