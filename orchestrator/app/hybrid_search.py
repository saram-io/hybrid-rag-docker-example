import numpy as np
from rank_bm25 import BM25Okapi
from app.config import settings
from app.vector_store import get_table
from app.embedding import get_embeddings_batch

class HybridSearchEngine:
    def __init__(self):
        self.all_chunks = []
        self.bm25 = None
        self.ready = False

    def rebuild(self):
        table = get_table()
        if table is None:
            return
        df = table.to_pandas()
        self.all_chunks = df.to_dict("records")
        if self.all_chunks:
            tokenized = [r["text"].lower().split() for r in self.all_chunks]
            self.bm25 = BM25Okapi(tokenized)
        self.ready = True

    @property
    def chunk_count(self) -> int:
        return len(self.all_chunks)

    @property
    def document_count(self) -> int:
        return len(set(r["source"] for r in self.all_chunks))

    def vector_search(self, query_embedding, limit):
        table = get_table()
        if table is None:
            return []
        return table.search(query_embedding).metric("cosine").limit(limit).to_list()

    def keyword_search(self, query, limit):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:limit]
        return [{**self.all_chunks[i], "_bm25_score": float(scores[i])}
                for i in top_idx if scores[i] > 0]

    def hybrid_search(self, query, top_k=None):
        top_k = top_k or settings.top_k
        fetch_k = top_k * 3

        query_embedding = get_embeddings_batch([query])[0]
        vec_results = self.vector_search(query_embedding, fetch_k)
        kw_results = self.keyword_search(query, fetch_k)

        # RRF fusion
        rrf_scores = {}
        id_to_result = {}

        for rank, r in enumerate(vec_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            id_to_result[cid] = r

        for rank, r in enumerate(kw_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            if cid not in id_to_result:
                id_to_result[cid] = r

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        final = []
        for cid in sorted_ids:
            entry = id_to_result[cid].copy()
            entry["_rrf_score"] = round(rrf_scores[cid], 6)
            final.append(entry)
        return final

engine = HybridSearchEngine()
