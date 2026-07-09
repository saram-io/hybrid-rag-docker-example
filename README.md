# Local Hybrid RAG Orchestration in Docker Compose

This is a complete, production-ready example of a hybrid Retrieval-Augmented Generation (RAG) pipeline (dense vector search + BM25 keyword matching) orchestrated using Docker Compose.

The architecture is fully self-hosted, keeping raw document text and embedding generation local.

## Architecture & Flow

```
┌────────────────────────── Docker Compose Network ───────────────────────────┐
│                                                                             │
│  ┌──────────────┐       ┌──────────────────┐       ┌──────────────────┐     │
│  │              │       │                  │       │                  │     │
│  │   RustFS     │◀──────│   RAG API        │──────▶│    Ollama        │     │
│  │  (S3 Store)  │       │  (Orchestrator)  │       │ qwen3-embedding  │     │
│  │   :9000      │       │    :8000         │       │    :11434        │     │
│  │              │       │                  │       │                  │     │
│  └──────────────┘       │  ┌────────────┐  │       └──────┬───────────┘     │
│                         │  │  LanceDB   │  │              │                 │
│                         │  │ (Vector +  │  │              │                 │
│                         │  │  BM25 FTS) │  │              │                 │
│                         │  └───────┬────┘  │       ┌──────▼───────────┐     │
│                         └──────────┴───────┘       │  Frontier Model  │     │
│                                    │               │ (OpenAI-compat)  │     │
│                                    │               │   via Ollama     │     │
│                                    │               └──────────────────┘     │
│                              ┌─────▼────┐                                   │
│                              │  Client  │                                   │
│                              └──────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **RustFS (S3 store)**: S3-compatible lightweight local storage written in Rust. Used to upload and persist raw documents.
2. **Ollama (AI engines)**: Serves `qwen3-embedding:8b` for embedding generation locally and any frontier model (e.g. `qwen3:32b` or `qwen3:8b`) for generation.
3. **LanceDB (Vector Database)**: Embedded vector and full-text search DB running inside the orchestrator.
4. **RAG API (Orchestrator)**: Fast API service written in Python tying all components together.

## Project Structure

```
├── docker-compose.yml
├── .env
├── README.md
├── orchestrator/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── embedding.py
│   │   ├── document_processor.py
│   │   ├── vector_store.py
│   │   ├── hybrid_search.py
│   │   └── llm_client.py
│   └── scripts/
│       └── init-ollama.sh
└── data/
    └── sample/
        └── sop-cleanroom-01.txt
```

## Running the Project

### Prerequisites
- Docker and Docker Compose installed.

### Steps
1. **Start all services:**
   ```bash
   docker compose up -d --build
   ```
2. **Watch the logs of the orchestrator/model puller:**
   ```bash
   docker compose logs -f rag-orchestrator
   ```
   *Note: On first startup, the Ollama container will pull both the embedding model and the frontier model, which might take some time depending on your network connection.*

3. **Verify Health:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Upload a Document:**
   ```bash
   curl -X POST http://localhost:8000/api/upload \
     -F "file=@data/sample/sop-cleanroom-01.txt"
   ```

5. **Trigger Ingestion (chunk, embed, and index):**
   ```bash
   curl -X POST http://localhost:8000/api/ingest
   ```

6. **Query the Pipeline:**
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What happens if CR-MODULE-99 reports a low temperature?"}'
   ```


## NVIDIA GPU Acceleration (Optional)

To run embedding generation and generation workloads on your local NVIDIA GPU (e.g. GeForce RTX 4090), you must configure GPU passthrough:

### 1. Prerequisites (Host Machine)
1. **NVIDIA Drivers**: Ensure you have proprietary NVIDIA drivers installed (run `nvidia-smi` to verify).
2. **Standard Docker Engine (Docker CE)**: Install the native Docker Engine on the host instead of **Docker Desktop for Linux** (Docker Desktop runs inside a VM and does not support direct GPU passthrough on Linux).
3. **NVIDIA Container Toolkit**: Bridge the host's GPU and Docker. Install the toolkit and register it with the Docker runtime:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. **Permissions**: Ensure your user has access to standard Docker:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   # For immediate agent/daemon access:
   sudo chmod 666 /var/run/docker.sock
   ```

### 2. Configure `docker-compose.yml`
Ensure your `docker-compose.yml` contains the `deploy` block under the `ollama` service:
```yaml
  ollama:
    # ... other config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 3. Verification
Verify that Ollama binds to your GPU during startup:
```bash
docker compose logs ollama
```
Look for CUDA initialization logs:
`ollama  | ... msg="inference compute" id=0 library=CUDA name=CUDA0 description="NVIDIA GeForce RTX 4090" ...`

---

## Tuning Retrieval Weights
The retrieval balance is controlled by the `DENSE_WEIGHT` environment variable:
- `0.3`: 70% Keyword search, 30% Semantic search (best for codes/part numbers).
- `0.5`: Equal weight (general purpose mixed content).
- `0.7`: 70% Semantic search, 30% Keyword search (best for narrative docs, default).
- `1.0`: Pure vector search.
- `0.0`: Pure BM25 keyword search.
