FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && pip install --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY fast.py embedding_client.py ./
COPY graph ./graph/

ENV LLM_N_GPU_LAYERS=0
ENV LLM_MODEL_PATH=/app/models/qwen2-1_5b-instruct-q4_k_m.gguf
ENV LLM_N_CTX=2048
ENV LLM_GENERATE_MAX_TOKENS=1024

EXPOSE 8000

CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000"]
