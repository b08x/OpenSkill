# --- Stage 1: Base System ---
FROM python:3.11-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: ML Dependencies (The Heavy Layer) ---
# Installing torch and core ML libs here ensures this massive layer is cached and shared
FROM base AS ml-base
COPY requirements.txt .
RUN pip install torch numpy scipy && \
    pip install -r requirements.txt

# --- Stage 3: OpenSkill Library ---
# All ML-enabled services (Web, MCP, PoC) will branch from here
FROM ml-base AS library
COPY OpenSkillLib/ /app/OpenSkillLib/
RUN pip install -e /app/OpenSkillLib/src

# --- Stage 4: Web UI & MCP Runtime ---
FROM library AS runtime-app
COPY main.py mcp_openSkill.py /app/
COPY templates/ /app/templates/
RUN mkdir -p /app/skills_output
EXPOSE 8000 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Stage 5: PoC Runtime (Local Inference) ---
FROM library AS runtime-poc
# Install extra deps only needed for the PoC
RUN pip install transformers sentence-transformers accelerate bitsandbytes safetensors
COPY OpenSkill1.1/ /app/OpenSkill1.1/
# We keep the app in its subdir but set PythonPath if needed, 
# or just copy files to root as the old one did. 
# The previous Poc Dockerfile copied files to /app/
WORKDIR /app/OpenSkill1.1
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

# --- Stage 6: Benchmark Runtime ---
FROM base AS runtime-benchmark
RUN pip install requests
COPY SkillTesting/MemCollab/ /app/
ENV CRAFT_API_URL=http://openskill-web:8000/api/craft \
    RETRIEVE_API_URL=http://openskill-web:8000/api/retrieve
CMD ["python", "--version"]
