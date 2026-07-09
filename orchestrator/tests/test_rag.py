import os
import unittest
from unittest.mock import patch, MagicMock

# Set environment variables for testing before importing application modules
os.environ["LANCEDB_URI"] = "/tmp/lancedb_test"
os.environ["RUSTFS_BUCKET"] = "test-bucket"

# Also append current directory to path so imports work correctly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.document_processor import extract_text, chunk_text
from app.embedding import get_embedding, get_embeddings_batch
from app.vector_store import init_table, add_records, get_table
from app.hybrid_search import engine, HybridSearchEngine
from app.llm_client import generate_answer

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Clean up temp LanceDB directory if exists
        import shutil
        if os.path.exists("/tmp/lancedb_test"):
            shutil.rmtree("/tmp/lancedb_test")

    def tearDown(self):
        import shutil
        if os.path.exists("/tmp/lancedb_test"):
            shutil.rmtree("/tmp/lancedb_test")

    def test_document_processor_chunking(self):
        # Test basic text chunking
        text = "This is a sentence. " * 50  # ~1000 characters
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            # We allow chunk_size + overlap + 1 due to tutorial chunk_text implementation adding overlap characters
            self.assertTrue(len(chunk) <= 200 + 20 + 1, f"Chunk too large: {len(chunk)}")

    def test_document_processor_extraction(self):
        # Test basic text extraction
        txt_content = b"Hello World Text File Content"
        result = extract_text(txt_content, "test.txt")
        self.assertEqual(result, "Hello World Text File Content")

    @patch("requests.Session.post")
    @patch("requests.post")
    def test_embedding_generation(self, mock_post, mock_session_post):
        # Mock requests.post response for embedding
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
        mock_post.return_value = mock_response
        mock_session_post.return_value = mock_response

        emb = get_embedding("test text")
        self.assertEqual(emb, [0.1, 0.2, 0.3, 0.4])

        embs = get_embeddings_batch(["text1", "text2"])
        self.assertEqual(embs, [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])

    def test_vector_store_and_hybrid_search(self):
        # Initialize test table (dimension 4)
        init_table(dimension=4)

        # Mock embedding function in hybrid_search.py during this test
        with patch("app.hybrid_search.get_embeddings_batch") as mock_emb:
            mock_emb.return_value = [[0.1, 0.2, 0.3, 0.4]]

            # We use at least 3 documents to avoid standard BM25 Okapi IDF returning 0 for 1-in-2 match.
            # We also use non-zero vectors to avoid division by zero during cosine search.
            records = [
                {"chunk_id": "doc1#0", "text": "All biological facilities must operate at 4C", "source": "doc1.txt", "vector": [0.9, 0.1, 0.0, 0.0]},
                {"chunk_id": "doc2#0", "text": "Sensor reading array reports a deviation event", "source": "doc2.txt", "vector": [0.0, 0.9, 0.1, 0.0]},
                {"chunk_id": "dummy1", "text": "This is a dummy document one", "source": "dummy1.txt", "vector": [0.1, 0.1, 0.1, 0.1]},
                {"chunk_id": "dummy2", "text": "This is a dummy document two", "source": "dummy2.txt", "vector": [0.1, 0.1, 0.1, 0.1]},
            ]
            add_records(records)

            # Rebuild hybrid search engine index
            engine.rebuild()

            self.assertEqual(engine.chunk_count, 4)
            self.assertEqual(engine.document_count, 4)

            # Test vector search
            vec_results = engine.vector_search([0.9, 0.1, 0.0, 0.0], limit=1)
            self.assertEqual(len(vec_results), 1)
            self.assertEqual(vec_results[0]["chunk_id"], "doc1#0")

            # Test keyword search
            kw_results = engine.keyword_search("deviation", limit=1)
            self.assertEqual(len(kw_results), 1)
            self.assertEqual(kw_results[0]["chunk_id"], "doc2#0")

            # Test hybrid RRF fusion search
            hybrid_results = engine.hybrid_search("deviation", top_k=2)
            self.assertTrue(len(hybrid_results) > 0)
            self.assertIn("_rrf_score", hybrid_results[0])

    @patch("openai.resources.chat.completions.Completions.create")
    def test_llm_generation(self, mock_chat_create):
        # Mock OpenAI API call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mocked LLM Answer"
        mock_chat_create.return_value = mock_response

        answer = generate_answer("What is the temperature?", ["Source chunk 1"])
        self.assertEqual(answer, "Mocked LLM Answer")

if __name__ == "__main__":
    unittest.main()
