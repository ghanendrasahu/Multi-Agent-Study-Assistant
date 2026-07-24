"""
memory/vector_store.py
──────────────────────
Long-term vector memory backed by ChromaDB + sentence-transformers.

Concepts from document:
  - Long-term memory (vector store backed): retrieves semantically similar past notes
  - Episodic memory: each study session's output is stored as an episode
  - Memory retrieval with similarity search: cosine similarity via ChromaDB
  - Memory consolidation: only high-quality (score ≥ 0.75) notes are stored
  - Embedding models: sentence-transformers (free, runs locally)
  - ChromaDB architecture: persistent local storage, no API key needed

Storage layout:
  Each document in ChromaDB:
    - content: the full study note text
    - metadata: {topic, query, score, session_id, timestamp}
    - embedding: 384-dim vector from all-MiniLM-L6-v2
"""

from __future__ import annotations

import os
from typing import Optional

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "chroma_db")
COLLECTION_NAME = "study_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, fast, 384 dimensions


class VectorMemory:
    """
    Long-term semantic memory using ChromaDB.

    Key design decisions (from document):
      1. sentence-transformers for embeddings — free, no API calls
      2. ChromaDB persistent mode — survives restarts
      3. Cosine similarity (ChromaDB default) — scale-invariant
      4. n_results=3 at retrieval — bounded context window usage
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._available = False
        self._init()

    def _init(self):
        """Lazily initialise ChromaDB and embedding model."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            os.makedirs(self.persist_dir, exist_ok=True)

            self._client = chromadb.PersistentClient(path=self.persist_dir)

            self._embedding_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL
                )
            )

            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"}  # HNSW index, cosine distance
            )

            self._available = True
            print(f"[Memory] ChromaDB initialised at {self.persist_dir}")
            print(f"[Memory] Collection '{COLLECTION_NAME}' has "
                  f"{self._collection.count()} notes stored.")

        except ImportError as e:
            print(f"[Memory] ChromaDB/sentence-transformers not installed: {e}")
            print("[Memory] Falling back to in-memory dict store.")
            self._fallback_store: list[dict] = []
            self._available = False
        except Exception as e:
            print(f"[Memory] Init failed: {e}")
            self._fallback_store = []
            self._available = False

    def store(
        self,
        content: str,
        topic: str,
        query: str,
        score: float,
        session_id: str,
        timestamp: str,
    ) -> bool:
        """
        Store a study note in long-term memory.

        Only stores if score >= 0.65 (quality gate = memory consolidation strategy).
        Uses the query as the document ID — upserts to avoid exact duplicates.
        """
        if score < 0.65:
            print(f"[Memory] Score {score:.2f} below threshold — not storing.")
            return False

        doc_id = _make_id(query)

        if self._available and self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[{
                        "topic": topic,
                        "query": query,
                        "score": str(score),
                        "session_id": session_id,
                        "timestamp": timestamp,
                    }]
                )
                print(f"[Memory] Stored '{topic}' (score={score:.2f})")
                return True
            except Exception as e:
                print(f"[Memory] Store failed: {e}")
                return False
        else:
            # Fallback in-memory store
            self._fallback_store.append({
                "id": doc_id, "content": content,
                "topic": topic, "query": query,
                "score": score, "timestamp": timestamp
            })
            return True

    def retrieve(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Retrieve semantically similar study notes for a query.

        Returns a list of dicts: {content, topic, query, score}
        Returns [] if no relevant notes found.

        Uses cosine similarity via HNSW index (from the document section on HNSW).
        """
        if self._available and self._collection is not None:
            try:
                if self._collection.count() == 0:
                    return []

                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self._collection.count()),
                    include=["documents", "metadatas", "distances"]
                )

                retrieved = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]

                for doc, meta, dist in zip(docs, metas, dists):
                    # ChromaDB cosine: distance 0 = identical, 2 = opposite
                    # Convert to similarity: 1 - (dist / 2)
                    similarity = 1.0 - (dist / 2.0)
                    if similarity > 0.3:  # Similarity threshold filter
                        retrieved.append({
                            "content": doc,
                            "topic": meta.get("topic", ""),
                            "query": meta.get("query", ""),
                            "score": float(meta.get("score", 0)),
                            "similarity": similarity
                        })

                return retrieved

            except Exception as e:
                print(f"[Memory] Retrieval failed: {e}")
                return []
        else:
            # Fallback: linear scan (no embeddings)
            return [
                {"content": n["content"], "topic": n["topic"],
                 "query": n["query"], "score": n["score"], "similarity": 1.0}
                for n in self._fallback_store[-n_results:]
            ]

    def list_topics(self) -> list[str]:
        """List all topics stored in memory."""
        if self._available and self._collection is not None:
            try:
                results = self._collection.get(include=["metadatas"])
                return list({m.get("topic", "") for m in results["metadatas"]})
            except Exception:
                return []
        return [n["topic"] for n in getattr(self, "_fallback_store", [])]

    def count(self) -> int:
        """Return number of notes in memory."""
        if self._available and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                return 0
        return len(getattr(self, "_fallback_store", []))

    @property
    def is_available(self) -> bool:
        return self._available


def _make_id(query: str) -> str:
    """Create a stable document ID from a query string."""
    import hashlib
    return hashlib.md5(query.lower().strip().encode()).hexdigest()
