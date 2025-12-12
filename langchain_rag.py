# langchain_rag.py
"""
Small LangChain wrapper to run a RetrievalQA chain over your vector store.
This file expects one of your stores (store_persistent.DocStorePersistent or store_faiss.FaissStore)
to expose `chunk_texts`+`embeddings` (for persistent) or to expose search() (for Faiss store).
We provide two helper adapters below:
 - build_vectorstore_from_persistent: builds a LangChain FAISS vectorstore from persisted texts+embeddings
 - build_chain_from_faiss_store: builds a RetrievalQA chain that queries your FaissStore.search() method
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI        # optional: swap with other LLM wrappers
import numpy as np

# Choose the same sentence-transformers model used elsewhere
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default prompt (keeps it grounded + asks for citations)
_DEFAULT_PROMPT = """You are a regulatory compliance assistant. Use ONLY the provided context to answer the user's question.
Cite sources inline in the form [title | chunk_id]. If information is not present in the context, respond: "No supporting text found in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=_DEFAULT_PROMPT, input_variables=["context", "question"])

def build_vectorstore_from_persistent(persistent_store):
    """
    Given an instance of DocStorePersistent (store_persistent.DocStorePersistent),
    build a LangChain FAISS vectorstore. This will wrap embeddings/texts so LangChain can use them.
    """
    # Use HuggingFaceEmbeddings wrapper (same model family)
    hf = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

    # persistent_store.chunk_texts -> list[str]
    texts = persistent_store.chunk_texts
    if not texts:
        # empty store
        return None

    # FAISS.from_texts will call embeddings internally; but to avoid recomputing we can pass embeddings
    # If persistent_store has numpy embeddings matching SentenceTransformer dims, convert to list
    if hasattr(persistent_store, "embeddings") and persistent_store.embeddings is not None:
        # Convert stored numpy embeddings into the vectorstore directly (FAISS.from_texts doesn't accept precomputed embeddings easily)
        # Simpler: use FAISS.from_texts which will compute embeddings using HF wrapper
        vect = FAISS.from_texts(texts, hf)
    else:
        vect = FAISS.from_texts(texts, hf)
    return vect


def build_chain_using_vectorstore(vectorstore, llm=None, chain_type="stuff"):
    """
    Given a LangChain vectorstore, return a RetrievalQA chain.
    - llm: pass a LangChain-compatible LLM (e.g., OpenAI(...)) or leave None to instantiate a default.
    """
    if llm is None:
        # Default LLM: OpenAI text model (requires OPENAI_API_KEY env var). Replace with other LLM if you want.
        llm = OpenAI(temperature=0.0, max_tokens=512)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":6})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, return_source_documents=True)
    return qa


def run_rag_query_from_persistent(persistent_store, question, llm=None):
    """
    End-to-end helper: build vectorstore from persistent store and run RetrievalQA for the question.
    Returns dict: {'answer': ..., 'source_documents': [...]}
    """
    vect = build_vectorstore_from_persistent(persistent_store)
    if vect is None:
        return {"answer": "No documents indexed.", "sources": []}
    qa = build_chain_using_vectorstore(vect, llm=llm)
    res = qa({"query": question})
    # res contains 'result' and 'source_documents' keys
    return {"answer": res.get("result"), "source_documents": [{"page_content": d.page_content, "metadata": getattr(d, "metadata", {})} for d in res.get("source_documents", [])]}
