import os
import requests

# ==========================================
# CONFIGURATIONS
# ==========================================
RETRIEVE_URL = os.getenv("RETRIEVE_API_URL", "http://localhost:8000/api/retrieve")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# We use a WEAK model to prove that the Skill teaches it to get it right
TEST_MODEL = "meta-llama/llama-3.1-8b-instruct"

# ==========================================
# TARGET TASKS (New problems based on benchmarks)
# ==========================================
# If the Skill worked, it should help the model solve these new problems:
TARGET_TASKS = [
    {
        "name": "Target: Number Theory",
        "query": "Find the last two digits of $3^{1000}$. Explain step-by-step."
    },
    {
        "name": "Target: Dependent Probability",
        "query": "A standard deck of 52 cards. You draw 2 cards at random without replacement. What is the probability that both are Aces? Show your work."
    },
    {
        "name": "Target: Overlapping Intervals Trap",
        "query": "Write a Python function to merge overlapping intervals. Give me the code. I am testing it with this exact array: [[1, 10], [2, 6], [8, 12], [15, 18]]. Explain how your code handles the [2, 6] being completely inside [1, 10]."
    }
]


def call_model(system_prompt, user_prompt):
    """Auxiliary function to call OpenRouter"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3  # Low for reasoning
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        data = response.json()

        # CHECK IF THERE WAS AN API ERROR (e.g., no credits, rate limit, etc.)
        if "error" in data:
            return f"❌ OPENROUTER ERROR:\n{data['error']}"

        # Check if 'choices' actually exists before accessing
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return f"❌ Unexpected API response: {data}"

    except Exception as e:
        return f"❌ Request error: {str(e)}"


def main():
    if API_KEY == "PASTE_YOUR_OPENROUTER_KEY_HERE" or not API_KEY:
        print("❌ ERROR: Configure your OPENROUTER_API_KEY.")
        return

    print("🧪 Starting MemCollab Inference Test...")
    print(f"🤖 Test Model: {TEST_MODEL}\n")

    for task in TARGET_TASKS:
        print("=" * 60)
        print(f"🎯 TASK: {task['name']}")
        print(f"📝 Question: {task['query']}\n")

        # ---------------------------------------------------------
        # STEP 1: TEST WITHOUT SKILL (Vanilla Baseline)
        # ---------------------------------------------------------
        print("▶️ TEST 1: Model WITHOUT Skill (Vanilla)...")
        vanilla_answer = call_model(
            "You are a helpful AI solving problems.",
            task['query']
        )
        print(f"❌ Answer (Vanilla):\n{vanilla_answer[:300]}...\n")

        # ---------------------------------------------------------
        # STEP 2: SEARCH FOR SKILL IN YOUR BACKEND
        # ---------------------------------------------------------
        print("🔍 Searching for Relevant Skill in SkillCrafter (/api/retrieve)...")
        retrieve_resp = requests.post(RETRIEVE_URL, json={
            "query": task['query'],
            "api_key": API_KEY,
            "top_k": 1
        }).json()

        if not retrieve_resp.get("results"):
            print("⚠️ No skill found. Run the benchmark first!")
            continue

        best_skill = retrieve_resp["results"][0]["meta"]
        skill_json = best_skill.get("skill", {})

        # Formatting constraints to put in the prompt
        constraints = "\n".join([f"- {c}" for c in retrieve_resp["results"][0]["meta"].get("constraints", [])])
        if not constraints:  # Fallback if the structure is in the internal dict
            try:
                import json
                with open(f"skills_output/{best_skill['filename']}", "r") as f:
                    content = f.read()
                    constraints = "Use the structural constraints described in the model."
            except:
                pass

        print(f"✅ Skill Found: {best_skill['title']} (Score: {retrieve_resp['results'][0]['score']})")

        # ---------------------------------------------------------
        # STEP 3: TEST WITH SKILL (MemCollab Augmented)
        # ---------------------------------------------------------
        print("\n▶️ TEST 2: Model WITH Skill (Augmented)...")
        system_with_skill = f"""You are a reasoning agent. 
You must follow these retrieved reasoning rules to avoid common pitfalls:
{retrieve_resp['results'][0]['content']}
"""
        augmented_answer = call_model(system_with_skill, task['query'])
        print(f"✅ Answer (Augmented):\n{augmented_answer[:500]}...\n")


if __name__ == "__main__":
    main()