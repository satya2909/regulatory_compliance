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
- This module tries multiple import locations for ChatOpenAI to be resilient across LangChain packaging variants.
"""

import os
from typing import List, Dict, Any, Optional
# from store_faiss import FaissStore

# LangChain imports (robust across versions)
try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain.prompts import PromptTemplate
    # from langchain.retrievers.base import BaseRetriever
    from langchain.schema import BaseRetriever,Document

    # Prefer ChatOpenAI from langchain.chat_models, fallback to langchain_openai or classic OpenAI
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        try:
            # some installations provide a separate package `langchain-openai`
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except Exception:
            # final fallback: classic OpenAI wrapper
            from langchain.llms import OpenAI as ChatOpenAI  # type: ignore
    # BaseLanguageModel for typing
    try:
        from langchain.base_language import BaseLanguageModel
    except Exception:
        # older/newer versions might have alternate path
        from langchain.schema import BaseLanguageModel  # type: ignore
except Exception as e:
    raise ImportError(
        "Please install langchain and the OpenAI integration (pip install langchain langchain-openai) "
        "or check your langchain installation."
    ) from e

# Default prompt
_PROMPT_TEMPLATE = """You are a regulatory compliance assistant. Use ONLY the provided context below to answer the user's question.
Cite sources inline in the form [title | chunk_id]. If information is not present in the context, respond: "No supporting text found in the provided documents."

CONTEXT:
{context}

Question:
{question}

Answer:"""

prompt_template = PromptTemplate(template=_PROMPT_TEMPLATE, input_variables=["context", "question"])


class StoreRetriever(BaseRetriever):
    """
    Adapter that implements LangChain's retriever interface by delegating to your store.search(query, top_k).
    It returns langchain.schema.Document objects.
    """
    def __init__(self, store, k: int = 6):
        self.store = store
        self.k = k

    def get_relevant_documents(self, query: str):
        return self._get_relevant_documents(query)

    def _get_relevant_documents(self, query: str):
        results = self.store.search(query, top_k=self.k)

        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r["text"],
                    metadata=r.get("metadata", {})
                )
            )
        return docs

    # alias to support other LangChain versions that call this method name
    async def aget_relevant_documents(self, query: str, **kwargs):
        return self.get_relevant_documents(query, **kwargs)
    

import os
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import List
from langchain.schema import BaseRetriever,Document
# from langchain.retrievers import BaseRetriever

class FaissStoreRetriever(BaseRetriever):
    store: object
    k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self.store.search(query, top_k=self.k)

        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r["text"],
                    metadata={
                        "doc_id": r["doc_id"],
                        "title": r.get("title", ""),
                        "chunk_id": r.get("chunk_id"),
                        "score": r.get("score"),
                    },
                )
            )
        return docs


class GeminiLLM(LLM):
    temperature: float = 0.0

    @property
    def _llm_type(self):
        return "gemini"

    def _call(self, prompt, stop=None):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text

def _get_default_llm(temperature=0.0):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    return GeminiLLM(temperature=temperature)


def build_chain_from_store(store, llm = None, top_k: int = 2):
    """
    Build a LangChain RetrievalQA chain that uses the StoreRetriever.
    - store: your FaissStore (or any store implementing search(...)).
    - llm: optional LangChain LLM (BaseLanguageModel). If None, tries to obtain default via OPENAI_API_KEY.
    - top_k: how many docs to retrieve by default (passed to retriever at query time).
    Returns: a dict { "chain": RetrievalQA | None, "retriever": StoreRetriever, "llm": llm_or_none }
    """
    llm = _get_default_llm(temperature=0.0)

    retriever = FaissStoreRetriever(store=store, k=3)
    if llm is None:
        llm = _get_default_llm(temperature=0.0)
    if llm is None:
        # No LLM available — we'll still return the retriever so caller can fetch docs and perform a grounded fallback.
        return {"chain": None, "retriever": retriever, "llm": None}

    # Build RetrievalQA with our retriever. We use 'stuff' chain type (puts retrieved docs directly into prompt).
    # Note: chain_type_kwargs accepts a 'prompt' in many LangChain versions; keep as dict wrapper.
    qa = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa


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
        return {
            "answer": "LLM is not configured properly.",
            "source_documents": []
        }


    # We have a chain: run it (LangChain chain expects {"query": question})
    # Depending on LangChain version, RetrievalQA returns either 'result' or 'answer' as the top-level key.
    res = qa({"query": question})
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
