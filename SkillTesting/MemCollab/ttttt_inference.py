import os
import requests

# ==========================================
# CONFIGURAÇÕES
# ==========================================
RETRIEVE_URL = os.getenv("RETRIEVE_API_URL", "http://localhost:8000/api/retrieve")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Vamos usar um modelo FRACO para provar que a Skill ensina ele a acertar
TEST_MODEL = "meta-llama/llama-3.1-8b-instruct"

# ==========================================
# TARGET TASKS (Novos problemas baseados nos benchmarks)
# ==========================================
# Se a Skill funcionou, ela tem que ajudar o modelo a resolver estes problemas novos:
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
    """Função auxiliar para chamar a OpenRouter"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3  # Baixo para raciocínio
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        data = response.json()

        # VERIFICA SE DEU ERRO NA API (ex: sem créditos, rate limit, etc)
        if "error" in data:
            return f"❌ ERRO DA OPENROUTER:\n{data['error']}"

        # Verifica se 'choices' realmente existe antes de acessar
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return f"❌ Resposta inesperada da API: {data}"

    except Exception as e:
        return f"❌ Erro na requisição: {str(e)}"


def main():
    if API_KEY == "COLE_SUA_CHAVE_OPENROUTER_AQUI" or not API_KEY:
        print("❌ ERRO: Configure sua OPENROUTER_API_KEY.")
        return

    print("🧪 Iniciando Teste de Inferência do MemCollab...")
    print(f"🤖 Modelo de Teste: {TEST_MODEL}\n")

    for task in TARGET_TASKS:
        print("=" * 60)
        print(f"🎯 TAREFA: {task['name']}")
        print(f"📝 Pergunta: {task['query']}\n")

        # ---------------------------------------------------------
        # PASSO 1: TESTE SEM A SKILL (Vanilla Baseline)
        # ---------------------------------------------------------
        print("▶️ TESTE 1: Modelo SEM a Skill (Vanilla)...")
        vanilla_answer = call_model(
            "You are a helpful AI solving problems.",
            task['query']
        )
        print(f"❌ Resposta (Vanilla):\n{vanilla_answer[:300]}...\n")

        # ---------------------------------------------------------
        # PASSO 2: BUSCAR A SKILL NO SEU BACKEND
        # ---------------------------------------------------------
        print("🔍 Buscando Skill Relevante no SkillCrafter (/api/retrieve)...")
        retrieve_resp = requests.post(RETRIEVE_URL, json={
            "query": task['query'],
            "api_key": API_KEY,
            "top_k": 1
        }).json()

        if not retrieve_resp.get("results"):
            print("⚠️ Nenhuma skill encontrada. Rode o benchmark antes!")
            continue

        best_skill = retrieve_resp["results"][0]["meta"]
        skill_json = best_skill.get("skill", {})

        # Formatando as constraints para colocar no prompt
        constraints = "\n".join([f"- {c}" for c in retrieve_resp["results"][0]["meta"].get("constraints", [])])
        if not constraints:  # Fallback caso a estrutura esteja no dict interno
            try:
                import json
                with open(f"skills_output/{best_skill['filename']}", "r") as f:
                    content = f.read()
                    constraints = "Use as constraints estruturais descritas no modelo."
            except:
                pass

        print(f"✅ Skill Encontrada: {best_skill['title']} (Score: {retrieve_resp['results'][0]['score']})")

        # ---------------------------------------------------------
        # PASSO 3: TESTE COM A SKILL (MemCollab Augmented)
        # ---------------------------------------------------------
        print("\n▶️ TESTE 2: Modelo COM a Skill (Augmented)...")
        system_with_skill = f"""You are a reasoning agent. 
You must follow these retrieved reasoning rules to avoid common pitfalls:
{retrieve_resp['results'][0]['content']}
"""
        augmented_answer = call_model(system_with_skill, task['query'])
        print(f"✅ Resposta (Augmented):\n{augmented_answer[:500]}...\n")


if __name__ == "__main__":
    main()