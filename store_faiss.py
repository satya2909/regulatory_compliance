# store_faiss.py
"""
FAISS-backed DocStore.
- Uses sentence-transformers for embeddings (same model family).
- Stores metadata (docs & chunk_index) as JSON, FAISS index as a binary file.
- Exposes add_document, list_documents, get_document, search, delete_document, save, load, clear.

Notes:
- This uses an IndexFlatIP (inner product) index with normalized vectors (so inner-product == cosine similarity).
- For larger datasets, consider IndexIVFFlat or HNSW indices (more complex but faster for many vectors).
"""

import os
import uuid
import json
import threading
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.vectorstores import FAISS


# Default settings
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "faiss_persist")
_INDEX_PATH = os.path.join(_PERSIST_DIR, "faiss.index")
_META_PATH = os.path.join(_PERSIST_DIR, "meta.json")
_DIM = 384  # embedding dim for all-MiniLM-L6-v2

def _ensure_persist_dir():
    os.makedirs(_PERSIST_DIR, exist_ok=True)

class FaissStore:
    def __init__(self, model_name=_EMBEDDING_MODEL_NAME, dim=_DIM):
        _ensure_persist_dir()
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._lock = threading.Lock()

        # metadata
        # docs: mapping doc_id -> {doc_id, title, chunks: [{chunk_id, text, char_range, page}]}
        self.docs: Dict[str, Dict] = {}
        # chunk_index: list mapping vector-row-index -> (doc_id, chunk_id)
        self.chunk_index: List[Tuple[str, str]] = []

        # FAISS index
        self.index = None
        self._ensure_model()
        # try loading existing index & metadata
        self.load()
        from langchain.vectorstores import FAISS
        from langchain.embeddings.base import Embeddings

        def as_langchain_vectorstore(self, embedding_model: Embeddings):
            return FAISS(
                embedding_function=embedding_model,
                index=self.index,
                docstore=self.docstore,
                index_to_docstore_id=self.index_to_docstore_id,
            )
        def get_retriever(self):
            """
            Returns a LangChain-compatible retriever
            """
            return self.vectorstore.as_retriever()


    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def _init_faiss_index(self):
        # IndexFlatIP for normalized vectors (fast, simple, exact)
        # dim must match the model embeddings dimension
        self.index = faiss.IndexFlatIP(self.dim)
        # Set nprobe if using IVF indices later (not used for IndexFlat)
        return self.index

    def add_document(self, filename: str, full_text: str) -> Dict:
        """
        Chunk the document, compute embeddings, add to FAISS index, persist metadata.
        Returns the stored document metadata structure.
        """
        from utils import chunk_text
        chunks = chunk_text(full_text)
        doc_id = uuid.uuid4().hex
        doc_chunks = []
        texts = []

        for c in chunks:
            chunk_id = uuid.uuid4().hex
            meta = {"chunk_id": chunk_id, "text": c["text"], "char_range": c["char_range"], "page": c.get("page", None)}
            doc_chunks.append(meta)
            texts.append(c["text"])

        with self._lock:
            # add doc metadata
            self.docs[doc_id] = {"doc_id": doc_id, "title": filename, "chunks": doc_chunks}
            # compute embeddings for new chunks
            if texts:
                self._ensure_model()
                emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                emb = np.asarray(emb, dtype=np.float32)
                # normalize for inner product similarity
                faiss.normalize_L2(emb)

                if self.index is None:
                    # create new index with right dim
                    self._init_faiss_index()

                # add to faiss index
                try:
                    self.index.add(emb)
                except Exception as e:
                    # in rare cases, write a helpful error
                    raise RuntimeError(f"Failed to add embeddings to FAISS index: {e}")

                # append chunk_index entries for each new vector row
                start_row = len(self.chunk_index)
                for i, c in enumerate(doc_chunks):
                    self.chunk_index.append((doc_id, c["chunk_id"]))

            # save metadata persistently
            self.save()

        return self.docs[doc_id]

    def list_documents(self):
        return [{"doc_id": d["doc_id"], "title": d["title"], "n_chunks": len(d["chunks"])} for d in self.docs.values()]

    def get_document(self, doc_id: str):
        return self.docs.get(doc_id)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Embed the query, search FAISS, return top_k results with metadata and similarity score.
        Score returned is inner product in [-1,1] (cosine since vectors are normalized).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        self._ensure_model()
        q_emb = self._model.encode([query], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_emb)

        # search
        D, I = self.index.search(q_emb, top_k)
        D = D[0]
        I = I[0]

        results = []
        for score, idx in zip(D, I):
            if idx < 0:
                continue
            try:
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
                    "score": float(score),
                    "text": chunk.get("text", "")
                })
            except IndexError:
                # inconsistent metadata vs index size - skip
                continue
        return results

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove a document's metadata and (re)build the FAISS index from remaining docs.
        FAISS doesn't support easy delete for IndexFlat; for robust deletions you can use IndexIDMap
        or maintain mapping and rebuild index. Here we rebuild for correctness.
        """
        with self._lock:
            if doc_id not in self.docs:
                return False
            del self.docs[doc_id]
            # rebuild index from all remaining docs
            all_texts = []
            new_chunk_index = []
            for d_id, d in self.docs.items():
                for c in d["chunks"]:
                    all_texts.append(c["text"])
                    new_chunk_index.append((d_id, c["chunk_id"]))

            # recreate index
            if len(all_texts) == 0:
                # clear everything
                self.index = None
                self.chunk_index = []
            else:
                self._ensure_model()
                emb = self._model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
                emb = np.asarray(emb, dtype=np.float32)
                faiss.normalize_L2(emb)
                self._init_faiss_index()
                self.index.add(emb)
                self.chunk_index = new_chunk_index

            # persist
            self.save()
            return True

    def save(self):
        """Persist the FAISS index and metadata to disk."""
        _ensure_persist_dir()
        # save metadata (docs & chunk_index)
        meta = {"docs": self.docs, "chunk_index": self.chunk_index, "dim": self.dim}
        with open(_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # save faiss index if present
        if self.index is not None:
            try:
                faiss.write_index(self.index, _INDEX_PATH)
            except Exception as e:
                # if write fails, log but don't raise (to keep app running)
                print("WARNING: failed to write FAISS index to disk:", e)

    def load(self):
        """Load metadata and FAISS index from disk (if present)."""
        _ensure_persist_dir()
        # load meta
        if os.path.exists(_META_PATH):
            try:
                with open(_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.docs = meta.get("docs", {})
                self.chunk_index = meta.get("chunk_index", [])
                # if dim stored, use it
                self.dim = meta.get("dim", self.dim)
            except Exception as e:
                print("Failed to load meta.json:", e)

        # load index
        if os.path.exists(_INDEX_PATH):
            try:
                idx = faiss.read_index(_INDEX_PATH)
                self.index = idx
            except Exception as e:
                print("Failed to load FAISS index:", e)
                # try to initialize empty index
                self.index = None
        else:
            self.index = None

    def clear(self):
        """Delete all data and persisted files."""
        with self._lock:
            self.docs = {}
            self.chunk_index = []
            self.index = None
            # remove persisted files
            try:
                if os.path.exists(_INDEX_PATH):
                    os.remove(_INDEX_PATH)
                if os.path.exists(_META_PATH):
                    os.remove(_META_PATH)
            except Exception:
                pass
