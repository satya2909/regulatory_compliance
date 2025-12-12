# langchain_full_integration.py
"""
Full LangChain integration using your existing FaissStore (or any store with `search(query, top_k)`).
This avoids recomputing embeddings and uses the store's search results as the retrieval layer.

Usage:
    from langchain_full_integration import build_chain_from_store, run_chain
    chain = build_chain_from_store(store)   # store = FaissStore() or similar
    out = run_chain(chain, "What are KYC requirements?", top_k=6)
    print(out)

Notes:
- Requires `langchain` installed.
- By default, selects OpenAI Chat models if OPENAI_API_KEY is set. If not set, run_chain will return a grounded fallback using the retrieved context (no hallucination).
"""

import os
from typing import List, Dict, Any, Optional

# LangChain imports
try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    # LLM wrappers
    try:
        from langchain.chat_models import ChatOpenAI
    except Exception:
        from langchain.llms import OpenAI as ChatOpenAI  # type: ignore
    from langchain.base_language import BaseLanguageModel
except Exception as e:
    raise ImportError("Please install langchain (pip install langchain).") from e

# Default prompt
_PROMPT_TEMPLATE = """You are a regulatory compliance assistant. Use ONLY the provided context below to answer the user's question.
Cite sources inline in the form [title | chunk_id]. If information is not present in the context, respond: "No supporting text found in the provided documents."

CONTEXT:
{context}

Question:
{question}

Answer:"""

prompt_template = PromptTemplate(template=_PROMPT_TEMPLATE, input_variables=["context", "question"])


class StoreRetriever:
    """
    Adapter that implements LangChain's retriever interface by delegating to your store.search(query, top_k).
    It returns langchain.schema.Document objects.
    """
    def __init__(self, store, default_k: int = 6):
        """
        store: object implementing `search(query, top_k)` -> list[dict]
               and ideally chunk metadata as returned by your store (doc_id, chunk_id, text, score).
        """
        self.store = store
        self.default_k = default_k

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        LangChain calls this method to get documents. We accept search kwargs: top_k or k.
        """
        top_k = int(kwargs.get("k") or kwargs.get("top_k") or self.default_k)
        # delegate to store.search
        hits = []
        try:
            hits = self.store.search(query, top_k=top_k) or []
        except Exception:
            hits = []

        docs: List[Document] = []
        for h in hits:
            # hit expected keys: 'text' (or 'page_content'), 'score', 'doc_id', 'chunk_id', 'title', ...
            page_content = h.get("text") or h.get("page_content") or ""
            metadata = {}
            # normalize metadata if available
            if "metadata" in h and isinstance(h["metadata"], dict):
                metadata = h["metadata"]
            else:
                # attach store-level metadata if present
                for k in ("doc_id", "chunk_id", "title", "score"):
                    if k in h:
                        metadata[k] = h[k]
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    # alias to support other LangChain versions that call this method name
    async def aget_relevant_documents(self, query: str, **kwargs):
        return self.get_relevant_documents(query, **kwargs)


def _get_default_llm(temperature: float = 0.0) -> Optional[BaseLanguageModel]:
    """
    Return a LangChain LLM if environment configured (OpenAI), otherwise None.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            # ChatOpenAI wrapper will pick model via OPENAI_API_KEY / env or default
            return ChatOpenAI(temperature=temperature)
        except Exception:
            # fallback to plain OpenAI LLM
            from langchain.llms import OpenAI
            return OpenAI(temperature=temperature)
    return None


def build_chain_from_store(store, llm: Optional[BaseLanguageModel] = None, top_k: int = 6):
    """
    Build a LangChain RetrievalQA chain that uses the StoreRetriever.
    - store: your FaissStore (or any store implementing search(...)).
    - llm: optional LangChain LLM (BaseLanguageModel). If None, tries to obtain default via OPENAI_API_KEY.
    - top_k: how many docs to retrieve by default (passed to retriever at query time).
    Returns: a dict { "chain": RetrievalQA | None, "retriever": StoreRetriever, "llm": llm_or_none }
    """
    retriever = StoreRetriever(store, default_k=top_k)
    if llm is None:
        llm = _get_default_llm(temperature=0.0)
    if llm is None:
        # No LLM available — we'll still return the retriever so caller can fetch docs and perform a grounded fallback.
        return {"chain": None, "retriever": retriever, "llm": None}

    # Build RetrievalQA with our retriever. We use 'stuff' chain type (puts retrieved docs directly into prompt).
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt_template})
    return {"chain": qa, "retriever": retriever, "llm": llm}


def run_chain(chain_obj: Dict[str, Any], question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the chain (if built) or return a grounded fallback from the retriever results.
    chain_obj is the dict returned by build_chain_from_store.
    Returns:
      {
        "answer": str,
        "source_documents": [ { "page_content": ..., "metadata": {...} } ],
        "grounded_fallback": bool  # True if we returned concatenated context because no LLM
      }
    """
    retriever: StoreRetriever = chain_obj["retriever"]
    qa = chain_obj.get("chain")
    llm = chain_obj.get("llm")

    # fetch documents using retriever (so we can always return context if no LLM)
    search_kwargs = {}
    if top_k is not None:
        search_kwargs["k"] = top_k
    docs = retriever.get_relevant_documents(question, **search_kwargs)

    # If no LLM, return grounded fallback (concatenated context)
    if qa is None or llm is None:
        # create grounded fallback
        if not docs:
            return {"answer": "No documents indexed.", "source_documents": [], "grounded_fallback": True}
        # concatenate a limited amount of context to keep prompt size reasonable
        parts = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            title = meta.get("title") or meta.get("doc_id") or f"doc_{i}"
            chunk_id = meta.get("chunk_id") or meta.get("id") or str(i)
            parts.append(f"[{i}] {title} | {chunk_id}\n{d.page_content}")
        context = "\n---\n".join(parts[:10])  # cap number of chunks in fallback
        answer = "[LLM not configured — returning concatenated context as grounded answer]\n\n" + context
        # normalize docs into simple dicts
        srcs = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        return {"answer": answer, "source_documents": srcs, "grounded_fallback": True}

    # We have a chain: run it (LangChain chain expects {"query": question})
    res = qa({"query": question})
    # LangChain RetrievalQA returns keys: 'result' or 'answer' and 'source_documents'
    answer = res.get("result") or res.get("answer") or ""
    source_documents = res.get("source_documents") or res.get("source_documents", []) or []
    # normalize to list of dicts
    normalized = []
    for d in source_documents:
        try:
            page_content = getattr(d, "page_content", None) or (d.get("page_content") if isinstance(d, dict) else "")
            metadata = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
        except Exception:
            page_content = str(d)
            metadata = {}
        normalized.append({"page_content": page_content, "metadata": metadata})
    return {"answer": answer, "source_documents": normalized, "grounded_fallback": False}
