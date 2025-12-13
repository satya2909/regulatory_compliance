# langchain_integration.py
import os
from typing import Dict, Any, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from langchain_groq import ChatGroq


# -----------------------------
# Prompt (grounded, strict)
# -----------------------------
PROMPT_TEMPLATE = """
You are a regulatory compliance assistant.

Answer the question using the provided context.
The wording in the document may differ from the question.
If the information is clearly implied or explicitly stated, answer YES/NO and explain briefly.

If the information is truly absent, say:
"Not found in the provided document. - integration"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (be concise and factual):
"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# -----------------------------
# FAISS Retriever Adapter
# -----------------------------
class FaissStoreRetriever(BaseRetriever):
    store: object
    k: int = 4

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self.store.search(query, top_k=self.k)

        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r["text"],
                    metadata={
                        "doc_id": r.get("doc_id"),
                        "title": r.get("title", ""),
                        "chunk_id": r.get("chunk_id"),
                        "score": r.get("score"),
                    },
                )
            )
        return docs

# ------------------------------------------------------------

# -----------------------------
# LLM (Groq – FREE)
# -----------------------------
def _get_default_llm(temperature: float = 0.0):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=512,
        timeout=60,
    )


# -----------------------------
# Build RetrievalQA Chain
# -----------------------------
def build_chain_from_store(store, top_k: int = 6):
    llm = _get_default_llm()

    retriever = FaissStoreRetriever(store=store, k=top_k)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )


# ------------------------------------------------------------


# -----------------------------
# Run Query
# -----------------------------
def run_chain(chain, question: str) -> Dict[str, Any]:
    try:
        result = chain({"query": question})

        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])

        if not answer:
            answer = "Not found in the provided document."

        return {
            "answer": answer,
            "source_documents": [
                {
                    "text": d.page_content,
                    "metadata": d.metadata,
                }
                for d in sources
            ],
        }

    except Exception as e:
        return {
            "answer": "LLM error. Please try again.",
            "error": str(e),
            "source_documents": [],
        }
