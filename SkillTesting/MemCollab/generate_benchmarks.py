import os
import time
import requests

# ==========================================
# CONFIGURAÇÕES
# ==========================================
API_URL = os.getenv("CRAFT_API_URL", "http://localhost:8000/api/craft")

# Pegue a chave do ambiente ou cole diretamente aqui
API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Recomendação: 8B para errar (Weak) e R1/Minimax para acertar (Strong)
WEAK_MODEL = "meta-llama/llama-3.1-8b-instruct"
STRONG_MODEL = "anthropic/claude-sonnet-4.6"

# ==========================================
# LISTA DE TAREFAS (BENCHMARKS)
# ==========================================
BENCHMARK_TASKS = [
    {
        "title": "Dependent Probability Trap",
        "task": "A bag contains 5 red balls and 3 blue balls. You draw 3 balls one by one WITHOUT replacement. What is the probability that you draw exactly 2 red balls and 1 blue ball? Show your full step-by-step reasoning."
    },
    {
        "title": "Number Theory Edge Case",
        "task": "Find the last two digits of $7^{2024}$. Explain your reasoning step-by-step."
    },
    {
        "title": "Overlapping Intervals Bug",
        "task": "Write a Python function to merge all overlapping intervals in a list of intervals (e.g., [[1,3],[2,6],[8,10],[15,18]]). The input might be unsorted. Explain your logic and how you handle edge cases where one interval completely swallows another."
    },
    {
        "title": "Binary Search Infinite Loop",
        "task": "Identify the bug in this Binary Search implementation and provide the corrected code. Explain why it fails:\ndef binary_search(arr, target):\n  low, high = 0, len(arr) - 1\n  while low <= high:\n    mid = (low + high) // 2\n    if arr[mid] == target: return mid\n    elif arr[mid] < target: low = mid\n    else: high = mid\n  return -1"
    },
    {
        "title": "Multi-Constraint Scheduling",
        "task": "Schedule 5 meetings (A, B, C, D, E) into 3 rooms (Room 1, Room 2, Room 3) between 9 AM and 11 AM. Each meeting is 1 hour long.\nConstraints:\n1. A must happen before C.\n2. B and D cannot be in the same room.\n3. E requires a projector (only Room 1 has one).\n4. A and B must happen at the same time.\nProvide a valid schedule and explain how constraints are satisfied."
    },
    {
        "title": "Logic Puzzle: Zebra Variant",
        "task": "Three friends (Alice, Bob, Charlie) have different pets (Dog, Cat, Bird) and drive different colored cars (Red, Blue, Green).\n- Alice is allergic to feathers.\n- The person with the Cat drives a Green car.\n- Bob drives a Red car.\n- Charlie hates dogs.\nWho owns which pet and drives which car? Show your deduction matrix step-by-step."
    }
]


def main():
    if API_KEY == "COLE_SUA_CHAVE_OPENROUTER_AQUI" or not API_KEY:
        print("❌ ERRO: Por favor, configure sua OPENROUTER_API_KEY no script ou nas variáveis de ambiente.")
        return

    print("🚀 Iniciando geração em massa de Skills (Benchmark MemCollab)...")
    print(f"🔗 URL Alvo: {API_URL}")
    print(f"🤖 Weak Model:  {WEAK_MODEL}")
    print(f"🧠 Strong Model: {STRONG_MODEL}")
    print(f"📦 Total de Tarefas: {len(BENCHMARK_TASKS)}\n")

    for index, item in enumerate(BENCHMARK_TASKS, start=1):
        print("-" * 50)
        print(f"⏳ Processando [{index}/{len(BENCHMARK_TASKS)}]: {item['title']}")

        payload = {
            "task": item["task"],
            "api_key": API_KEY,
            "weak_model": WEAK_MODEL,
            "strong_model": STRONG_MODEL
        }

        start_time = time.time()

        try:
            # Note: O timeout é longo pois modelos de raciocínio podem demorar até 2 minutos
            response = requests.post(API_URL, json=payload, timeout=240)
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Sucesso! (Tempo: {elapsed_time:.1f}s)")
                print(f"   🏷️  Título da Skill: {data.get('title')}")
                print(f"   📁 Categoria: {data.get('category')} > {data.get('subcategory')}")
                print(f"   📄 Arquivo salvo: {data.get('filename')}")
            else:
                print(f"❌ Falha no Backend (Tempo: {elapsed_time:.1f}s)")
                print(f"   Status Code: {response.status_code}")
                print(f"   Erro: {response.text}")

        except requests.exceptions.ConnectionError:
            print("❌ ERRO: Não foi possível conectar ao backend. O 'main.py' está rodando na porta 8000?")
            break
        except requests.exceptions.Timeout:
            print("❌ ERRO: Timeout. A requisição demorou mais de 4 minutos.")
        except Exception as e:
            print(f"❌ ERRO Inesperado: {str(e)}")

        # Pausa de 5 segundos entre as requisições para evitar rate limit na OpenRouter
        if index < len(BENCHMARK_TASKS):
            print("⏳ Pausando 5 segundos antes da próxima requisição...")
            time.sleep(5)

    print("\n🎉 Processo de Benchmark finalizado! Verifique sua pasta 'skills_output'.")


if __name__ == "__main__":
    # Garante que a biblioteca requests esteja instalada
    try:
        import requests
    except ImportError:
        print("A biblioteca 'requests' não está instalada. Execute: pip install requests")
        exit(1)

    main()