"""
Full LangChain integration using Gemini (Google Generative AI)
with your existing FaissStore (or any store with `search(query, top_k)`).

Key properties:
- No re-embedding (uses store.search)
- Gemini-only (no OpenAI clutter)
- Grounded fallback if GOOGLE_API_KEY is missing
- Compatible with latest LangChain APIs

Usage:
    from langchain_full_integration import build_chain_from_store, run_chain
    chain_obj = build_chain_from_store(store)
    out = run_chain(chain_obj, "What are KYC requirements?", top_k=6)
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

# ------------------------------------------------

# -------- Prompt Template (grounded & citation-safe) --------
_PROMPT_TEMPLATE = """You are a regulatory compliance assistant.
Use ONLY the provided context below to answer the user's question.

Rules:
- Do NOT use any outside knowledge.
- Every factual statement must be supported by the context.
- Cite sources inline as [title | chunk_id].
- If the answer is not found in the context, respond exactly:
  "No supporting text found in the provided documents."

CONTEXT:
{context}

Question:
{question}

Answer:
"""

prompt_template = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ------------------------------------------------------------


class StoreRetriever(BaseRetriever):
    """
    LangChain Retriever adapter that delegates retrieval
    to your existing store.search(query, top_k).
    """

    def __init__(self, store, default_k: int = 6):
        self.store = store
        self.default_k = default_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            hits = self.store.search(query, top_k=self.default_k) or []
        except Exception:
            hits = []

        docs: List[Document] = []
        for h in hits:
            page_content = h.get("text") or ""
            metadata = {
                "doc_id": h.get("doc_id"),
                "chunk_id": h.get("chunk_id"),
                "title": h.get("title"),
                "page": h.get("page"),
                "section": h.get("section"),
                "score": h.get("score"),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

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

# ------------------------------------------------------------

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

def _get_gemini_llm(temperature: float = 0.0) -> Optional[ChatGoogleGenerativeAI]:
    """
    Returns Gemini LLM if GOOGLE_API_KEY is set, else None.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temperature,
        convert_system_message_to_human=True
    )


# ------------------------------------------------------------


def build_chain_from_store(store, top_k: int = 6) -> Dict[str, Any]:
    """
    Builds a RetrievalQA chain using Gemini + StoreRetriever.

    Returns:
      {
        "chain": RetrievalQA | None,
        "retriever": StoreRetriever,
        "llm": Gemini LLM | None
      }
    """
    retriever = StoreRetriever(store, default_k=top_k)
    llm = _get_gemini_llm(temperature=0.0)

    if llm is None:
        return {
            "chain": None,
            "retriever": retriever,
            "llm": None
        }

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return {
        "chain": qa_chain,
        "retriever": retriever,
        "llm": llm
    }


# ------------------------------------------------------------


def run_chain(
    chain_obj: Dict[str, Any],
    question: str,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes the RAG chain or returns a grounded fallback.

    Output:
      {
        "answer": str,
        "source_documents": [{page_content, metadata}],
        "grounded_fallback": bool
      }
    """

    retriever: StoreRetriever = chain_obj["retriever"]
    qa = chain_obj.get("chain")

    if top_k is not None:
        retriever.default_k = top_k

    docs = retriever.get_relevant_documents(question)

    # ---- Grounded fallback (NO Gemini) ----
    if qa is None:
        if not docs:
            return {
                "answer": "No documents indexed.",
                "source_documents": [],
                "grounded_fallback": True
            }

        parts = []
        for i, d in enumerate(docs[:8], start=1):
            meta = d.metadata or {}
            title = meta.get("title") or meta.get("doc_id") or f"doc_{i}"
            chunk_id = meta.get("chunk_id") or str(i)
            parts.append(f"[{i}] {title} | {chunk_id}\n{d.page_content}")

        context = "\n---\n".join(parts)

        return {
            "answer": "[Gemini not configured — returning grounded context]\n\n" + context,
            "source_documents": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in docs
            ],
            "grounded_fallback": True
        }

    # ---- Gemini-powered RAG ----
    res = qa({"query": question})

    answer = res.get("result", "")
    src_docs = res.get("source_documents", [])

    normalized_sources = [
        {
            "page_content": d.page_content,
            "metadata": d.metadata
        }
        for d in src_docs
    ]

    return {
        "answer": answer,
        "source_documents": normalized_sources,
        "grounded_fallback": False
    }
