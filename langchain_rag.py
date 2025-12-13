# langchain_rag.py
"""
LangChain RAG helper for the RegTech skeleton.

Features:
- Builds a LangChain FAISS vectorstore from your store's texts (uses HuggingFaceEmbeddings).
- Creates a RetrievalQA chain (LangChain) and runs queries, returning:
    { "answer": <str>, "source_documents": [ {page_content, metadata}, ... ] }

Usage (example):
    from store_persistent import DocStorePersistent
    from langchain_rag import RAGRunner

    store = DocStorePersistent()
    runner = RAGRunner(store)
    out = runner.run(question="What are the KYC requirements?", top_k=6)
    print(out["answer"])
    print(out["source_documents"])
"""

from typing import Optional, List, Dict, Any
import os

# LangChain imports
try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    # For LLM: prefer ChatOpenAI, fallback to OpenAI wrapper
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception as e:
    raise ImportError("Please install langchain and its dependencies. pip install langchain") from e

# Default HF embedding model (should match your sentence-transformers model family)
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default prompt template for grounded answers
_PROMPT_TEMPLATE = """You are a regulatory compliance assistant. Use ONLY the provided context below to answer the user's question.
Cite sources inline using the form [title | chunk_id]. If the information is not present in the context, say "No supporting text found in the provided documents."

CONTEXT:
{context}

Question: {question}

Answer:"""

_prompt = PromptTemplate(template=_PROMPT_TEMPLATE, input_variables=["context", "question"])


class _FallbackLLM:
    """Simple fallback 'LLM' used when no cloud LLM is configured.
    It returns the concatenated context (so responses remain grounded and non-hallucinated).
    """

    def __init__(self, notice: str = "[LLM not configured — returning concatenated context]"):
        self.notice = notice

    def __call__(self, prompt: str, **kwargs):
        # attempt to extract the CONTEXT block like in the prompt template
        marker = "CONTEXT:"
        if marker in prompt:
            try:
                ctx = prompt.split(marker, 1)[1].split("Question:", 1)[0].strip()
                return f"{self.notice}\\n\\n{ctx}"
            except Exception:
                pass
        return self.notice + " (context not found)"


class RAGRunner:
    """
    RAGRunner wraps:
      - building a LangChain FAISS vectorstore from a store (store must expose chunk_texts or list_documents)
      - creating a RetrievalQA chain with an LLM (OpenAI by default if OPENAI_API_KEY present)
      - running queries and returning results with source_documents
    """

    def __init__(self, store, hf_embedding_model: str = HF_EMBEDDING_MODEL):
        """
        store: your store object (store_persistent.DocStorePersistent or store_faiss.FaissStore or similar).
               It must expose:
                 - attribute `chunk_texts` (list[str]) OR a method `list_documents()` + `get_document(doc_id)`
               and optionally:
                 - attribute `chunk_metadata` / chunk ids in the stored doc chunks for better citations.
        """
        self.store = store
        self.hf_embedding_model = hf_embedding_model
        # cache vectorstore once built
        self._vectorstore = None

    def _build_vectorstore_from_store(self):
        """
        Build a LangChain FAISS vectorstore from store.chunk_texts (recomputes embeddings using HuggingFaceEmbeddings).
        For small/medium datasets this is fine; for very large datasets consider constructing FAISS directly from precomputed embeddings.
        """
        if self._vectorstore is not None:
            return self._vectorstore

        # Try to obtain texts
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        # Preferred: store exposes chunk_texts list (simple)
        if hasattr(self.store, "chunk_texts") and isinstance(self.store.chunk_texts, list) and len(self.store.chunk_texts) > 0:
            texts = list(self.store.chunk_texts)
            # Try to construct minimal metadata: if store has chunk_index or docs mapping, attach doc_id/chunk_id
            if hasattr(self.store, "chunk_index"):
                for i, pair in enumerate(self.store.chunk_index):
                    meta = {}
                    try:
                        doc_id, chunk_id = pair
                        meta = {"doc_id": doc_id, "chunk_id": chunk_id}
                    except Exception:
                        meta = {"idx": i}
                    metadatas.append(meta)
        else:
            # fallback: iterate documents and collect their chunks
            if hasattr(self.store, "list_documents"):
                for d in self.store.list_documents():
                    doc_id = d.get("doc_id")
                    doc = self.store.get_document(doc_id)
                    if not doc or "chunks" not in doc:
                        continue
                    for c in doc["chunks"]:
                        texts.append(c.get("text", ""))
                        # build metadata from chunk fields if present
                        meta = {
                            "doc_id": doc_id,
                            "title": doc.get("title"),
                            "chunk_id": c.get("chunk_id"),
                            "page": c.get("page")
                        }
                        metadatas.append(meta)

        if not texts:
            # nothing to index
            self._vectorstore = None
            return None

        # Build LangChain HuggingFaceEmbeddings (wrapper uses sentence-transformers under the hood)
        hf = HuggingFaceEmbeddings(model_name=self.hf_embedding_model)

        # Create FAISS vectorstore from texts and metadata
        # This will compute embeddings via hf and build a FAISS index internally.
        # For small demos it's fine; if you already have precomputed embeddings, see advanced path (not implemented here).
        self._vectorstore = FAISS.from_texts(texts, hf, metadatas=metadatas)
        return self._vectorstore

    def _get_llm(self, temperature: float = 0.0):
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=temperature,
            google_api_key=gemini_key
        )


    def run(self, question: str, top_k: int = 6, return_source_documents: bool = True) -> Dict[str, Any]:
        """
        Run the RetrievalQA chain:
        Returns:
            {
                "answer": <str>,
                "source_documents": [ { "page_content": ..., "metadata": { ... } }, ... ],
                "raw": <langchain raw output if any>
            }
        """
        # Build or reuse vectorstore
        vectorstore = self._build_vectorstore_from_store()
        if vectorstore is None:
            return {"answer": "No documents indexed.", "source_documents": []}

        # Build retriever with desired k
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

        # Get LLM
        llm = self._get_llm(temperature=0.0)

        # Build chain - using 'stuff' chain_type (puts context into prompt). Could use 'map_reduce' for large docs.
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=return_source_documents, chain_type_kwargs={"prompt": _prompt})

        # Execute
        res = qa({"query": question})
        # res typically contains 'result' and 'source_documents' keys
        answer = res.get("result") or res.get("answer") or ""
        src_docs = res.get("source_documents") or res.get("source_documents", [])

        # Normalize source_documents into plain dicts (page_content + metadata)
        normalized_sources = []
        for d in src_docs:
            # LangChain's Document shape: Document(page_content=..., metadata={...})
            try:
                page_content = getattr(d, "page_content", None) or d.get("page_content", "")
                metadata = getattr(d, "metadata", None) or d.get("metadata", {})
            except Exception:
                # fallback if it's a plain dict
                page_content = d.get("page_content", "") if isinstance(d, dict) else str(d)
                metadata = d.get("metadata", {}) if isinstance(d, dict) else {}
            normalized_sources.append({"page_content": page_content, "metadata": metadata})

        return {"answer": answer, "source_documents": normalized_sources, "raw": res}
