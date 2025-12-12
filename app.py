from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
from store import DocStore
from utils import extract_text_from_file, chunk_text
from store import DocStorePersistent  # or from store_faiss import FaissStore
from langchain_rag import run_rag_query_from_persistent
# instantiate persistent store instead of the in-memory one:


app = Flask(__name__)
store = DocStore()
store = DocStorePersistent()
INDEX_HTML = """
<!doctype html>
<title>RegTech RAG - Minimal Upload</title>
<h1>Upload a PDF or TXT</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p>Use /query to ask questions (POST JSON: {'question': '...'}).</p>
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
    return jsonify({"status":"ok", "doc_id": doc["doc_id"], "chunks": len(doc["chunks"])})

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
    results = store.search(question, top_k=5)
    return jsonify({"question": question, "results": results})

@app.route('/rag_query', methods=['POST'])
def rag_query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error":"provide JSON with 'question' field"}), 400
    question = data['question']
    res = run_rag_query_from_persistent(store, question)
    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
