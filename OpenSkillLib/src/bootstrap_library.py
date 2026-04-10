import subprocess
import re
import os
import sys

# --- CONFIGURATION ---
MY_OPENROUTER_KEY = "sk-or-v1-"

TASKS = [
    "Implement a Connection Pool for PostgreSQL in Python using psycopg2",
    "Design a Circuit Breaker pattern for microservices to handle cascading failures",
    "Secure a REST API using JWT (JSON Web Tokens) with asymmetric RS256 keys",
    "Optimize SQL queries using Composite Indexes and Explain Analyze",
    "Implement an Exponential Backoff retry strategy for failed HTTP requests",
    "Configure Kubernetes Liveness and Readiness probes for a high-availability service",
    "Manage distributed transactions using the Saga Pattern in a microservices architecture",
    "Implement Blue-Green deployment strategy using Nginx as a load balancer",
    "Detect and prevent Memory Leaks in long-running Python asyncio processes",
    "Implement Role-Based Access Control (RBAC) in a FastAPI application"
]

def run_command(command):
    print(f"\n> Executing: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=os.environ
        )

        if result.returncode != 0:
            print(f"WARNING: Command returned an error (code {result.returncode})")
            print(f"Stderr: {result.stderr}")

        return result.stdout if result.stdout else ""
    except Exception as e:
        print(f"CRITICAL ERROR while executing command: {e}")
        return ""

def main():
    if not MY_OPENROUTER_KEY:
        print("ERROR: Configure the OPENROUTER_API_KEY in the script!")
        return

    os.environ["OPENROUTER_API_KEY"] = MY_OPENROUTER_KEY

    skill_ids = []
    print("=== Starting OpenSkill Library Creation (UTF-8 Mode) ===")

    for i, task in enumerate(TASKS):
        print(f"\n" + "=" * 40)
        print(f"--- Creating Skill {i + 1}/{len(TASKS)} ---")
        print(f"Task: {task}")

        # 1. openskill create
        cmd_create = f'openskill create "{task}" --api-key {MY_OPENROUTER_KEY}'
        output = run_command(cmd_create)

        # 2. Extract ID
        match = re.search(r"ID:.*?([a-f0-9]{8})", output, re.IGNORECASE)

        if match:
            skill_id = match.group(1)
            skill_ids.append(skill_id)
            print(f"SUCCESS: ID {skill_id} created.")

            # 3. Generate Local Embedding (FIXED: Removed --skill-id)
            print(f"Generating local embedding (384d)...")
            run_command(f"openskill embed {skill_id} --local")
        else:
            print("WARNING: Could not capture the ID automatically.")

    # 4. GNN Refinement
    print("\n" + "=" * 50)
    print("FINAL STAGE: Refining library with GNN...")
    print("=" * 50)
    run_command("openskill build-graph --use-gnn")

    print("\n=== PROCESS COMPLETED ===")
    print(f"Processed skills: {len(skill_ids)}")

if __name__ == "__main__":
    main()