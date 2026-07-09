import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import boto3
from botocore.client import Config as BotoConfig

from app.config import settings
from app.document_processor import extract_text, chunk_text
from app.embedding import get_embeddings_batch, get_embedding
from app.vector_store import init_table, add_records
from app.hybrid_search import engine
from app.llm_client import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client(
    "s3",
    endpoint_url=f"http://{settings.rustfs_endpoint}",
    aws_access_key_id=settings.rustfs_access_key,
    aws_secret_access_key=settings.rustfs_secret_key,
    config=BotoConfig(signature_version="s3v4"),
    region_name="us-east-1",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    from botocore.exceptions import ClientError
    
    # Ensure bucket exists (wait/retry for RustFS to be ready)
    max_retries = 30
    for i in range(max_retries):
        try:
            s3.head_bucket(Bucket=settings.rustfs_bucket)
            break
        except Exception as e:
            if isinstance(e, ClientError):
                status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status_code == 404:
                    try:
                        s3.create_bucket(Bucket=settings.rustfs_bucket)
                        break
                    except Exception as create_err:
                        if i == max_retries - 1:
                            raise create_err
            if i == max_retries - 1:
                logger.error("Failed to connect to RustFS at %s", settings.rustfs_endpoint)
                raise e
            logger.warning("RustFS not ready yet (attempt %d/%d). Retrying in 2s...", i + 1, max_retries)
            time.sleep(2)

    # Wait/retry for embedding model to be ready in Ollama
    dim = None
    max_ollama_retries = 180
    for i in range(max_ollama_retries):
        try:
            dim = len(get_embedding("dimension probe"))
            break
        except Exception as e:
            if i == max_ollama_retries - 1:
                logger.error("Failed to connect to Ollama or model %s not ready", settings.embedding_model)
                raise e
            logger.warning("Ollama or embedding model not ready (attempt %d/%d). Retrying in 5s...", i + 1, max_ollama_retries)
            time.sleep(5)

    init_table(dim)
    engine.rebuild()
    logger.info("Ready. %d chunks indexed. Embedding dim=%d.", engine.chunk_count, dim)
    yield

app = FastAPI(title="Hybrid RAG Orchestrator", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    top_k: int = settings.top_k

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    timings: dict

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents": engine.document_count,
        "chunks": engine.chunk_count,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.frontier_model,
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    s3.put_object(Bucket=settings.rustfs_bucket, Key=file.filename, Body=data)
    return {"filename": file.filename, "size": len(data)}

@app.post("/api/ingest")
async def ingest():
    objects = s3.list_objects_v2(Bucket=settings.rustfs_bucket).get("Contents", [])
    total_chunks = 0
    files_processed = 0

    for obj in objects:
        key = obj["Key"]
        ext = key.rsplit(".", 1)[-1].lower()
        if ext not in {"pdf", "txt", "md", "docx"}:
            continue

        raw = s3.get_object(Bucket=settings.rustfs_bucket, Key=key)["Body"].read()
        text = extract_text(raw, key)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)

        if not chunks:
            continue

        embeddings = get_embeddings_batch(chunks)
        records = [
            {"chunk_id": f"{key}#{i}", "text": c, "source": key, "vector": e}
            for i, (c, e) in enumerate(zip(chunks, embeddings))
        ]
        add_records(records)
        total_chunks += len(records)
        files_processed += 1

    engine.rebuild()
    return {"files": files_processed, "chunks": total_chunks}

@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if engine.chunk_count == 0:
        raise HTTPException(400, "No documents ingested. Upload and run /api/ingest first.")

    t0 = time.time()
    results = engine.hybrid_search(req.query, req.top_k)
    search_ms = round((time.time() - t0) * 1000)

    context_chunks = [r["text"] for r in results]
    sources = [
        {"source": r["source"], "text": r["text"][:200], "rrf_score": r.get("_rrf_score", 0)}
        for r in results
    ]

    t0 = time.time()
    answer = generate_answer(req.query, context_chunks)
    llm_ms = round((time.time() - t0) * 1000)

    return QueryResponse(
        answer=answer,
        sources=sources,
        timings={"search_ms": search_ms, "llm_ms": llm_ms, "total_ms": search_ms + llm_ms},
    )
