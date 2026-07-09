import requests
from app.config import settings

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{settings.ollama_host}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": text.strip()},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    embeddings = []
    with requests.Session() as session:
        for text in texts:
            resp = session.post(
                f"{settings.ollama_host}/api/embeddings",
                json={"model": settings.embedding_model, "prompt": text.strip()},
                timeout=120
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
    return embeddings
