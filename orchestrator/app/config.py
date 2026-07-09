import os
from dataclasses import dataclass

@dataclass
class Config:
    # RustFS
    rustfs_endpoint: str = os.getenv("RUSTFS_ENDPOINT", "rustfs:9000")
    rustfs_access_key: str = os.getenv("RUSTFS_ACCESS_KEY", "admin")
    rustfs_secret_key: str = os.getenv("RUSTFS_SECRET_KEY", "admin123456")
    rustfs_bucket: str = os.getenv("RUSTFS_BUCKET", "documents")

    # Ollama
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")

    # Frontier Model (OpenAI-compatible)
    frontier_url: str = os.getenv("FRONTIER_MODEL_URL", "http://ollama:11434/v1")
    frontier_model: str = os.getenv("FRONTIER_MODEL_NAME", "qwen3:32b")
    frontier_api_key: str = os.getenv("FRONTIER_API_KEY", "ollama")

    # LanceDB
    lancedb_uri: str = os.getenv("LANCEDB_URI", "/app/data/lancedb")
    table_name: str = "documents"

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.7"))
    rrf_k: int = 60

settings = Config()
