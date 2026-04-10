import asyncio
import os

from openskill import TurboQuantizer
from openskill.client import OpenSkillClient
from openskill.llm.openrouter import OpenRouterProvider

# Put your OpenRouter key here
MY_KEY = "sk-or-v1"
os.environ["OPENROUTER_API_KEY"] = MY_KEY


async def main():
    llm = OpenRouterProvider(api_key=MY_KEY)
    # Increase dimension here to ensure Client doesn't force 384d
    client = OpenSkillClient(llm=llm, skill_dir="./skills_output")
    client._quantizer = TurboQuantizer(dimension=1536)

    print("\n--- 1. CRAFTING SKILL ---")
    skill_meta = await client.craft("Calculate the Fibonacci sequence optimally using memoization")

    # FORCE GRAPH RECONSTRUCTION AFTER CRAFT
    print("--- Building Knowledge Graph ---")
    from openskill.core.graph import register_skill_in_graph
    all_metas = await client.store.list_skills()
    metas_dict = {m.id: m.to_dict() for m in all_metas}

    # Register skill in the graph
    await register_skill_in_graph(
        skill_id=skill_meta.id,
        meta=skill_meta,
        all_metas=metas_dict,
        store=client.store
    )

    graph_data = client.store.get_graph()
    print(f"Graph updated with {len(graph_data.nodes)} nodes.")

    print("\n--- 2. EXECUTING QUEST ---")
    # Now that the graph exists, search will work
    response = await client.execute_quest("Find the 100th number in the Fibonacci sequence")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())