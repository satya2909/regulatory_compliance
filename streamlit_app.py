import streamlit as st
import requests
import io
import json
from typing import Any, Dict, List

# === CONFIG ===
# Change this to your Flask backend address if needed:
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://192.168.144.223:7860/")
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload"
QUERY_ENDPOINT = f"{BACKEND_URL}/query"
RAG_QUERY_ENDPOINT = f"{BACKEND_URL}/rag_query"

st.set_page_config(page_title="Docs RAG UI", layout="wide")

# === Helpers ===
def post_file_to_backend(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    files = {"file": (filename, io.BytesIO(file_bytes))}
    resp = requests.post(UPLOAD_ENDPOINT, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()

def post_json(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

def show_sources(sources: List[Dict[str, Any]]):
    for i, s in enumerate(sources, start=1):
        st.markdown(f"**Source {i} — score: {s.get('score', 'N/A')}, id: {s.get('id', s.get('doc_id', ''))}**")
        # text could be in 'page_content' or 'text' or 'content' depending on your pipeline
        content = s.get("page_content") or s.get("text") or s.get("content")
        if content:
            with st.expander("Preview source text", expanded=False):
                st.write(content[:3000] + ("..." if len(content) > 3000 else ""))
        # show meta if present
        meta = s.get("metadata") or s.get("meta")
        if meta:
            st.write("Metadata:", meta)
        st.markdown("---")


# === UI ===
st.title("📚 Document Upload & RAG — Streamlit Frontend")
st.caption("A simple UI that talks to your Flask backend (`/upload`, `/query`, `/rag_query`).")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"], accept_multiple_files=False)
    if uploaded_file:
        filename = uploaded_file.name
        st.write("Filename:", filename)
        if st.button("Upload to backend"):
            try:
                with st.spinner("Uploading..."):
                    result = post_file_to_backend(uploaded_file.getvalue(), filename)
                st.success("Uploaded!")
                st.json(result)
                # store doc_id in session for convenience
                if "doc_id" in result:
                    st.session_state["doc_id"] = result["doc_id"]
                if "n_chunks" in result:
                    st.session_state["n_chunks"] = result["n_chunks"]
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown("**Stored document**")
    st.write("doc_id:", st.session_state.get("doc_id", "—"))
    if "n_chunks" in st.session_state:
        st.write("n_chunks:", st.session_state["n_chunks"])

    st.markdown("---")
    st.header("Simple retrieval query")
    q_text = st.text_area("Question (simple retrieval):", value="", height=80)
    if st.button("Ask (simple retrieval)"):
        if not q_text.strip():
            st.warning("Please type a question.")
        else:
            payload = {"question": q_text}
            if st.session_state.get("doc_id"):
                payload["doc_id"] = st.session_state["doc_id"]
            try:
                with st.spinner("Querying..."):
                    resp = post_json(QUERY_ENDPOINT, payload)
                st.success("Got response")
                st.markdown("**Response JSON**")
                st.json(resp)
                if "sources" in resp:
                    st.markdown("**Sources**")
                    show_sources(resp["sources"])
            except Exception as e:
                st.error(f"Query failed: {e}")

with col2:
    st.header("RAG (LLM-backed) query")
    rag_q = st.text_area("Question (RAG):", value="", height=80)
    top_k = st.slider("top_k (how many top chunks to retrieve)", min_value=1, max_value=20, value=6)
    threshold = st.slider("threshold (score threshold 0.0–1.0)", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    use_doc = st.checkbox("Pass doc_id to backend (if present)", value=True)

    if st.button("Ask (RAG)"):
        if not rag_q.strip():
            st.warning("Please type a question.")
        else:
            payload = {"question": rag_q, "top_k": top_k, "threshold": threshold}
            if use_doc and st.session_state.get("doc_id"):
                payload["doc_id"] = st.session_state["doc_id"]
            try:
                with st.spinner("Running RAG... (may take a few seconds)"):
                    resp = post_json(RAG_QUERY_ENDPOINT, payload)
                st.success("RAG finished")
                st.markdown("**Response JSON**")
                st.json(resp)
                # Many RAG endpoints return 'answer' and 'source_documents' or 'sources'
                answer = resp.get("answer") or resp.get("result") or resp.get("response") or resp.get("output")
                if answer:
                    st.markdown("### ✅ Answer")
                    st.write(answer)
                # Show sources if present
                sources = resp.get("source_documents") or resp.get("sources") or resp.get("source_docs")
                if sources:
                    st.markdown("### 📎 Source documents")
                    show_sources(sources)
            except Exception as e:
                st.error(f"RAG query failed: {e}")

st.markdown("---")
st.header("Backend diagnostics")
st.write("Current BACKEND_URL:", BACKEND_URL)
if st.button("Ping backend"):
    try:
        r = requests.get(BACKEND_URL, timeout=10)
        st.write("Status:", r.status_code)
        # show text/html homepage
        st.text(r.text[:5000])
    except Exception as e:
        st.error(f"Ping failed: {e}")

st.markdown("**Raw endpoints used**")
st.code(f"UPLOAD -> {UPLOAD_ENDPOINT}\nQUERY -> {QUERY_ENDPOINT}\nRAG -> {RAG_QUERY_ENDPOINT}")
