"""
High-performance PubMed abstract retriever using:
- FAISS ANN indexing (replaces brute-force linear scan)
- sentence-transformers with ONNX Runtime (replaces Ollama/raw PyTorch)

Performance at 3M+ vectors on M1 Mac (Docker):
- Query embedding: ~5-15ms (ONNX) vs ~30-80ms (PyTorch) vs ~200ms+ (Ollama HTTP)
- Vector search: ~1-5ms (FAISS ANN) vs 1-5 minutes (brute-force)
- Total query latency: <50ms vs minutes

Index types supported:
- flat: Exact search, no training. Good for <500K vectors.
- ivf: IVF with flat quantizer. Good for 500K-10M vectors.
- ivfpq: IVF with product quantization. 10M+ vectors, 16x memory reduction.
- ivfsq: IVF with scalar quantization. 10M+ vectors, 4x memory reduction.
"""

import faiss
import numpy as np
import sqlite3
import logging
import time

logger = logging.getLogger(__name__)


class AbstractRetrieverFaiss:
    def __init__(self, index_path, id_map_path, db_file,
                 model_name="nomic-ai/nomic-embed-text-v1.5",
                 use_onnx=True,
                 nprobe=32):
        """
        Args:
            index_path: Path to the FAISS index file (.faiss)
            id_map_path: Path to the numpy file mapping FAISS IDs to PMIDs (.npy)
            db_file: Path to the SQLite database with article metadata
            model_name: HuggingFace model name for query embeddings
            use_onnx: Use ONNX Runtime for faster embedding inference
            nprobe: Number of IVF clusters to search (higher = better recall, slower).
                    Only applies to IVF-based indices. Ignored for flat indices.
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.db_file = db_file
        self.model_name = model_name
        self.nprobe = nprobe

        self._load_model(use_onnx)
        self._load_index()
        self._load_id_map()
        self.connection = self._connect_db()

    def _load_model(self, use_onnx):
        """Load embedding model, preferring ONNX Runtime for speed."""
        from sentence_transformers import SentenceTransformer

        if use_onnx:
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    backend="onnx",
                    model_kwargs={"provider": "CPUExecutionProvider"},
                    trust_remote_code=True
                )
                self._backend = "onnx"
                logger.info("Loaded embedding model with ONNX Runtime backend")
                return
            except Exception as e:
                logger.warning(f"ONNX backend unavailable ({e}), falling back to PyTorch")

        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True
        )
        self._backend = "pytorch"
        logger.info("Loaded embedding model with PyTorch backend")

    def _load_index(self):
        """Load FAISS index from disk."""
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        # Set search parameters for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe

        logger.info(f"FAISS index loaded: {self.index.ntotal:,} vectors")

    def _load_id_map(self):
        """Load FAISS sequential ID -> PMID mapping."""
        self.id_map = np.load(self.id_map_path)
        logger.info(f"ID map loaded: {len(self.id_map):,} entries")

    def _connect_db(self):
        connection = sqlite3.connect(self.db_file)
        return connection

    def _fetch_document_info(self, pmids):
        cursor = self.connection.cursor()
        # Use parameterized query to prevent SQL injection
        placeholders = ",".join("?" * len(pmids))
        query = (
            f"SELECT pmid, title, authors, abstract, publication_year "
            f"FROM articles WHERE pmid IN ({placeholders})"
        )
        cursor.execute(query, [str(p) for p in pmids])
        rows = cursor.fetchall()
        cursor.close()

        # Build a lookup map to preserve the ranking order from FAISS
        doc_map = {}
        for row in rows:
            doc_map[str(row[0])] = {
                'pmid': row[0],
                'title': row[1],
                'authors': row[2],
                'abstract': row[3],
                'publication_year': row[4]
            }

        # Return in the same order as the input pmids (preserves FAISS ranking)
        return [doc_map[str(p)] for p in pmids if str(p) in doc_map]

    def embed_query(self, query):
        """Embed a query string. Returns L2-normalized float32 vector.

        Uses the "search_query: " prefix required by the Nomic model
        to distinguish queries from documents at embedding time.
        """
        embedding = self.model.encode(
            "search_query: " + query,
            normalize_embeddings=True
        )
        return embedding.astype('float32')

    def search(self, query, top_k=10):
        """
        Search for the most similar PubMed abstracts.

        Args:
            query: Natural language search query
            top_k: Number of results to return

        Returns:
            pmids: array of PubMed IDs
            similarities: array of cosine similarity scores (descending)
            documents: list of document metadata dicts (same order as pmids)
        """
        t0 = time.perf_counter()
        query_vector = self.embed_query(query).reshape(1, -1)
        t_embed = time.perf_counter() - t0

        t0 = time.perf_counter()
        # FAISS returns (distances, indices) arrays of shape (1, top_k)
        # For IndexFlatIP / METRIC_INNER_PRODUCT, distances = cosine similarities
        # (because vectors are L2-normalized)
        similarities, indices = self.index.search(query_vector, top_k)
        t_search = time.perf_counter() - t0

        similarities = similarities[0]
        indices = indices[0]

        # Filter out invalid results (FAISS returns -1 for unfilled slots)
        valid = indices >= 0
        indices = indices[valid]
        similarities = similarities[valid]

        # Map FAISS sequential IDs to PMIDs
        pmids = self.id_map[indices]

        t0 = time.perf_counter()
        documents = self._fetch_document_info(pmids)
        t_fetch = time.perf_counter() - t0

        total_ms = (t_embed + t_search + t_fetch) * 1000
        logger.info(
            f"Search timing: embed={t_embed*1000:.1f}ms, "
            f"faiss={t_search*1000:.1f}ms, "
            f"db={t_fetch*1000:.1f}ms, "
            f"total={total_ms:.1f}ms"
        )

        return pmids, similarities, documents
