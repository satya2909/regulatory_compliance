import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import json
import os

# Choose a small, fast model for demos: all-MiniLM-L6-v2
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class DocStore:
    def __init__(self, model_name=_EMBEDDING_MODEL_NAME):
        self.docs = {}  # doc_id -> metadata & chunks
        self.chunk_texts = []  # flattened list of chunk texts
        self.chunk_index = []  # list of (doc_id, chunk_id)
        self.embeddings = None  # numpy array shape (n_chunks, dim), dtype=np.float32
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        # lazy model load
        self._ensure_model()

    def _ensure_model(self):
        if self._model is None:
            # load on first use
            self._model = SentenceTransformer(self.model_name)

    def add_document(self, filename, full_text):
        """
        Adds a document, chunks it, computes embeddings for new chunks and updates index.
        Returns stored document metadata.
        """
        doc_id = uuid.uuid4().hex
        from utils import chunk_text  # import here to avoid circular issues
        chunks = chunk_text(full_text)
        doc_chunks = []

        new_texts = []
        new_index_entries = []

        for c in chunks:
            chunk_id = uuid.uuid4().hex
            meta = {"chunk_id": chunk_id, "text": c["text"], "char_range": c["char_range"], "page": c.get("page", None)}
            doc_chunks.append(meta)
            # store for global flattened arrays
            new_texts.append(c["text"])
            new_index_entries.append((doc_id, chunk_id))

        with self._lock:
            # update docs
            self.docs[doc_id] = {"doc_id": doc_id, "title": filename, "chunks": doc_chunks}

            # extend flattened lists
            start = len(self.chunk_texts)
            self.chunk_texts.extend(new_texts)
            self.chunk_index.extend(new_index_entries)

            # compute embeddings for the new texts and append to embeddings matrix
            if len(new_texts) > 0:
                new_emb = self._model.encode(new_texts, show_progress_bar=False, convert_to_numpy=True)
                # ensure float32 for memory/perf and consistent shape
                new_emb = np.asarray(new_emb, dtype=np.float32)
                if self.embeddings is None:
                    self.embeddings = new_emb
                else:
                    # use concatenate (slightly faster than vstack in repeated ops)
                    self.embeddings = np.concatenate([self.embeddings, new_emb], axis=0)

        return self.docs[doc_id]

    def list_documents(self):
        return [{"doc_id": d["doc_id"], "title": d["title"], "n_chunks": len(d["chunks"])} for d in self.docs.values()]

    def get_document(self, doc_id):
        return self.docs.get(doc_id)

    def delete_document(self, doc_id):
        """Remove a document and its chunks from the index (rebuilds flattened arrays)."""
        with self._lock:
            if doc_id not in self.docs:
                return False
            # remove doc
            del self.docs[doc_id]
            # rebuild flattened lists from remaining docs (safe but O(n))
            all_texts = []
            all_index = []
            for d_id, d in self.docs.items():
                for c in d["chunks"]:
                    all_texts.append(c["text"])
                    all_index.append((d_id, c["chunk_id"]))
            # recompute embeddings for all_texts (costly) - or if you had persisted embeddings, you'd remove rows instead
            if len(all_texts) == 0:
                self.chunk_texts = []
                self.chunk_index = []
                self.embeddings = None
                return True
            # encode again (keeps consistent)
            new_embs = self._model.encode(all_texts, show_progress_bar=False, convert_to_numpy=True)
            self.chunk_texts = all_texts
            self.chunk_index = all_index
            self.embeddings = np.asarray(new_embs, dtype=np.float32)
            return True

    def search(self, query, top_k=5):
        """
        Embedding-based cosine similarity search over chunks.
        Returns top_k results with doc metadata and score.
        """
        if self.embeddings is None or len(self.chunk_texts) == 0:
            return []

        # ensure model loaded
        self._ensure_model()

        # embed query
        q_emb = self._model.encode([query], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        # compute cosine similarity
        scores = cosine_similarity(q_emb, self.embeddings).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if idx < 0 or idx >= len(self.chunk_index):
                continue
            score = float(scores[idx])
            doc_id, chunk_id = self.chunk_index[idx]
            doc = self.docs.get(doc_id)
            if not doc:
                continue
            chunk = next((c for c in doc["chunks"] if c["chunk_id"] == chunk_id), None)
            if chunk is None:
                continue
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "chunk_id": chunk_id,
                "score": score,
                "text": chunk.get("text", "")
            })
        return results

    def save(self, path):
        """Persist docs, chunk_texts, chunk_index, and embeddings to `path` folder."""
        os.makedirs(path, exist_ok=True)
        with self._lock:
            with open(os.path.join(path, "docs.json"), "w", encoding="utf-8") as f:
                json.dump(self.docs, f, ensure_ascii=False, indent=2)
            with open(os.path.join(path, "chunk_texts.json"), "w", encoding="utf-8") as f:
                json.dump(self.chunk_texts, f, ensure_ascii=False)
            with open(os.path.join(path, "chunk_index.json"), "w", encoding="utf-8") as f:
                json.dump(self.chunk_index, f, ensure_ascii=False)
            if self.embeddings is not None:
                np.save(os.path.join(path, "embeddings.npy"), self.embeddings)

    def load(self, path):
        """Load persisted store from `path` folder if present."""
        with self._lock:
            try:
                docs_p = os.path.join(path, "docs.json")
                ct_p = os.path.join(path, "chunk_texts.json")
                ci_p = os.path.join(path, "chunk_index.json")
                emb_p = os.path.join(path, "embeddings.npy")
                if os.path.exists(docs_p):
                    with open(docs_p, "r", encoding="utf-8") as f:
                        self.docs = json.load(f)
                if os.path.exists(ct_p):
                    with open(ct_p, "r", encoding="utf-8") as f:
                        self.chunk_texts = json.load(f)
                if os.path.exists(ci_p):
                    with open(ci_p, "r", encoding="utf-8") as f:
                        self.chunk_index = json.load(f)
                if os.path.exists(emb_p):
                    self.embeddings = np.load(emb_p)
            except Exception as e:
                # don't crash on load errors; keep store empty
                print("Failed to load persisted store:", e)

    def clear(self):
        """Clear all in-memory data (useful for tests)."""
        with self._lock:
            self.docs = {}
            self.chunk_texts = []
            self.chunk_index = []
            self.embeddings = None
