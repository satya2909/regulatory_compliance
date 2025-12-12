# app.py (updated)
from flask import Flask, request, jsonify, render_template_string
import os
from utils import extract_text_from_file
from store_faiss import FaissStore
from langchain_integration import build_chain_from_store, run_chain
from verify import verify_answer

app = Flask(__name__)

# Choose the FAISS-backed store (make sure store_faiss.py is present)
store = FaissStore()

# Build LangChain chain object once at startup (reuses your store's FAISS index)
chain_obj = build_chain_from_store(store)  # uses OPENAI_API_KEY if present

INDEX_HTML = """
<!doctype html>
<title>RegTech RAG - Minimal Upload</title>
<h1>Upload a PDF or TXT</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p>Use /query to ask questions (POST JSON: {'question': '...'}).</p>
<p>Use /rag_query to get LLM-backed answers (POST JSON: {'question':'...', 'top_k':6}).</p>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'no file part', 400
    f = request.files['file']
    if f.filename == '':
        return 'no selected file', 400
    filename = f.filename
    content = f.read()
    text = extract_text_from_file(content, filename)
    if not text:
        return 'failed to extract text', 400
    doc = store.add_document(filename, text)
    return jsonify({"status":"ok", "doc_id": doc["doc_id"], "n_chunks": len(doc["chunks"])})

@app.route('/docs', methods=['GET'])
def list_docs():
    return jsonify(store.list_documents())

@app.route('/docs/<doc_id>/chunks', methods=['GET'])
def doc_chunks(doc_id):
    doc = store.get_document(doc_id)
    if not doc:
        return 'not found', 404
    return jsonify({"doc_id": doc_id, "chunks": doc["chunks"]})

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error":"provide JSON with 'question' field"}), 400
    question = data['question']
    top_k = int(data.get('top_k', 5))
    results = store.search(question, top_k=top_k)
    return jsonify({"question": question, "results": results})

@app.route('/rag_query', methods=['POST'])
def rag_query():
    """
    Runs the LangChain retrieval chain (via build_chain_from_store/run_chain) and then verifies grounding.
    Request JSON:
      { "question": "...", "top_k": 6, "threshold": 0.65, "verify_top_k": 3 }
    Response:
      { "answer": ..., "sources": [...], "verification": {...} }
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error":"provide JSON with 'question' field"}), 400

    question = data['question']
    top_k = int(data.get('top_k', 6))

    # Run the LangChain-backed retrieval/QA (this will use your FaissStore via the retriever adapter)
    try:
        out = run_chain(chain_obj, question, top_k=top_k)
    except Exception as e:
        return jsonify({"error": "RAG chain failed", "details": str(e)}), 500

    # Normalize outputs
    answer = out.get("answer", "")
    source_documents = out.get("source_documents", [])  # list of {page_content, metadata}
    grounded_fallback = bool(out.get("grounded_fallback", False))

    # Map LangChain source_documents to verifier-friendly format: {"text": ..., "score": ... , "metadata": ...}
    sources_for_verify = []
    for sd in source_documents:
        page_content = sd.get("page_content") if isinstance(sd, dict) else ""
        metadata = sd.get("metadata") if isinstance(sd, dict) else {}
        # try to extract a score if present in metadata
        score = metadata.get("score") if isinstance(metadata, dict) else None
        sources_for_verify.append({"text": page_content, "metadata": metadata, "score": score})

    # Run verification: prefer verifying against the retrieved sources first, then fall back to store.search
    try:
        verify_report = verify_answer(
            answer,
            store,
            initial_chunks=sources_for_verify,
            threshold=float(data.get("threshold", 0.65)),
            top_k=int(data.get("verify_top_k", 3))
        )
    except Exception as e:
        verify_report = {"error": "verification failed", "details": str(e)}

    response = {
        "answer": answer,
        "grounded_fallback": grounded_fallback,
        "sources": sources_for_verify,
        "verification": verify_report
    }
    return jsonify(response)

if __name__ == '__main__':
    # FaissStore.load() is called in FaissStore.__init__ if implemented, so persisted index is loaded automatically.
    app.run(host='0.0.0.0', port=7860, debug=True)
