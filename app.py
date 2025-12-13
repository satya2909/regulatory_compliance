# app.py — LangChain full integration + citation injection
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import requests
# from openai import OpenAI
from pdf_diff import build_pdf_diff_highlights
from utils import extract_text_from_file
from store_faiss import FaissStore
from langchain_integration import build_chain_from_store, run_chain
from cite import inject_citations
from version_compare import compare_documents, compare_texts
from utils import extract_text_from_file, chunk_text
from langchain_groq import ChatGroq
import re


retriever = None


app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'])

# Initialize store and LangChain chain object (retriever uses store.search)
store = FaissStore()
chain_obj = build_chain_from_store(store)  # will use OPENAI_API_KEY if present

INDEX_HTML = """
<!doctype html>
<title>RegTech RAG - Minimal Upload</title>
<h1>Upload a PDF or TXT</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p>Use /query to ask questions (POST JSON: {'question': '...'}).</p>
<p>Use /rag_query to get LLM-backed answers (POST JSON: {'question':'...', 'top_k':6, 'threshold':0.65}).</p>
"""
# model = genai.GenerativeModel("models/gemini-2.5-flash")
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_unique_texts(results):
    seen = set()
    unique_texts = []

    for r in results:
        text = r["text"].strip()

        # normalize whitespace
        text_norm = " ".join(text.split())

        if text_norm not in seen:
            seen.add(text_norm)
            unique_texts.append(text)

    return unique_texts

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"you are now",
    r"act as",
    r"system prompt",
]

def input_guardrail(user_query: str):
    query_lower = user_query.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_lower):
            raise ValueError("Potential prompt injection detected.")
    
    if len(user_query) > 500:
        raise ValueError("Query too long.")

def retrieval_guardrail(docs, min_score=0.5):
    if not docs:
        raise ValueError("No relevant documents found.")

    filtered_docs = [d for d in docs if d.metadata["score"] >= min_score]

    if not filtered_docs:
        raise ValueError("Low-confidence retrieval. Cannot answer reliably.")

    return filtered_docs

def hallucination_guardrail(answer, context):
    for sentence in answer.split("."):
        if sentence.strip() and sentence not in context:
            return False
    return True

def context_guardrail(context_chunks):
    if not context_chunks or len(context_chunks) == 0:
        return False

    total_text = " ".join(context_chunks)
    if len(total_text.strip()) < 200:
        return False

    return True


def generate_answer_with_groq(question: str, context_chunks: list[str]) -> str:
    try:
        input_guardrail(question)

        if not context_guardrail(context_chunks):
            return "Not found in the provided document - app.py."

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
        You are a regulatory compliance assistant.

        Answer the question using the provided context.
        The wording in the document may differ from the question.
        If the information is clearly implied or explicitly stated, answer YES/NO and explain briefly.

        If the information is truly absent, say:
        "Not found in the provided document - app.py."

        Document Context:
        {context_text}

        Question:
        {question}

        Answer:
        """

        response = llm.invoke(prompt)

        if not response or not response.content:
            return "Not found in the document."

        return response.content.strip()

    except Exception as e:
        print("Groq error:", e)
        return "LLM error. Please try again later."



@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/compare_versions', methods=['POST'])
def compare_versions():
    """
    Compare two versions. Two modes:
    1) Provide JSON: {"old_doc_id": "...", "new_doc_id": "...", "match_threshold":0.72, "modify_threshold":0.9}
    2) Provide multipart/form-data with files 'old_file' and 'new_file' (PDFs/TXT); optional thresholds in form fields.

    Returns JSON diff with added/removed/modified/unchanged lists and summary.
    """
    # mode 1: JSON doc ids
    if request.is_json:
        data = request.get_json()
        old_id = data.get("old_doc_id")
        new_id = data.get("new_doc_id")
        match_threshold = float(data.get("match_threshold", 0.72))
        modify_threshold = float(data.get("modify_threshold", 0.90))
        if not old_id or not new_id:
            return jsonify({"error": "provide old_doc_id and new_doc_id in JSON"}), 400
        old_doc = store.get_document(old_id)
        new_doc = store.get_document(new_id)
        if not old_doc or not new_doc:
            return jsonify({"error": "one or both doc_ids not found"}), 404

        res = compare_documents(old_doc, new_doc, match_threshold=match_threshold, modify_threshold=modify_threshold)
        return jsonify(res), 200

    # mode 2: file uploads
    if 'old_file' in request.files and 'new_file' in request.files:
        old_f = request.files['old_file']
        new_f = request.files['new_file']
        match_threshold = float(request.form.get("match_threshold", 0.72))
        modify_threshold = float(request.form.get("modify_threshold", 0.90))

        old_text = extract_text_from_file(old_f.read(), old_f.filename)
        new_text = extract_text_from_file(new_f.read(), new_f.filename)
        if old_text is None or new_text is None:
            return jsonify({"error": "failed to extract text from one or both files"}), 400

        # use compare_texts convenience wrapper
        res = compare_texts(old_text, new_text, match_threshold=match_threshold, modify_threshold=modify_threshold)
        return jsonify(res), 200

    return jsonify({"error": "send JSON with doc ids OR multipart with old_file + new_file"}), 400

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    text = extract_text_from_file(f.read(), f.filename)
    if not text:
        return jsonify({"error": "Failed to extract text"}), 400

    doc = store.add_document(f.filename, text)

    return jsonify({
        "status": "ok",
        "doc_id": doc["doc_id"],
        "n_chunks": len(doc["chunks"])
    })


@app.route('/docs', methods=['GET'])
def list_docs():
    return jsonify(store.list_documents())


@app.route('/docs/<doc_id>/chunks', methods=['GET'])
def doc_chunks(doc_id):
    doc = store.get_document(doc_id)
    if not doc:
        return 'not found', 404
    return jsonify({"doc_id": doc_id, "chunks": doc["chunks"]})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # 🔍 Step 1: search FAISS with scores
    results = store.search(question, top_k=8)

    if not results:
        return jsonify({
            "final_answer": "Not found in the document.",
            "results": []
        })

    # 🧠 Step 2: filter by similarity score
   # 🧠 Step 2: collect context (NO harsh score cutoff)
    context_chunks = []
    seen = set()

    for r in results:
        text = r["text"].strip()
        norm = " ".join(text.split())

        if norm not in seen:
            seen.add(norm)
            context_chunks.append(text)

    if not context_chunks:
        return jsonify({
            "final_answer": "Not found in the document.",
            "results": results
        })


    # 🤖 Step 3: ask Groq
    final_answer = generate_answer_with_groq(question, context_chunks)

    return jsonify({
        "final_answer": final_answer,
        "results": results
    })


@app.route("/pdf_diff", methods=["POST"])
def pdf_diff():
    """
    Compare two documents and return PDF diff highlights.

    JSON:
    {
      "old_doc_id": "...",
      "new_doc_id": "..."
    }
    """
    data = request.get_json()
    old_id = data.get("old_doc_id")
    new_id = data.get("new_doc_id")

    old_doc = store.get_document(old_id)
    new_doc = store.get_document(new_id)

    if not old_doc or not new_doc:
        return jsonify({"error": "Invalid doc_id(s)"}), 400

    compare_result = compare_documents(old_doc, new_doc)
    highlights = build_pdf_diff_highlights(compare_result)

    return jsonify({
        "old_doc_id": old_id,
        "new_doc_id": new_id,
        "summary": compare_result.get("summary"),
        "highlights": highlights
    })




@app.route('/rag_query', methods=['POST'])
def rag_query():
    """
    Runs the LangChain chain (or returns grounded fallback) then injects inline citations.
    Request JSON:
      { "question": "...", "top_k": 6, "threshold": 0.65 }
    Response JSON:
      {
        "cited_answer": "...",
        "raw_answer": "...",
        "sources": [...],
        "verification": { "total_sentences": N, "grounded_count": G, "sentences": [...] },
        "grounded_fallback": False
      }
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "provide JSON with 'question' field"}), 400

    question = data['question']
    top_k = int(data.get('top_k', 6))
    threshold = float(data.get('threshold', 0.65))
    verify_top_k = int(data.get('verify_top_k', 3))

    # Run the chain (returns answer + source_documents) — chain_obj was built at startup
    try:
        out = run_chain(chain_obj, question, top_k=top_k)
    except Exception as e:
        return jsonify({"error": "RAG chain failed", "details": str(e)}), 500

    # Extract results
    raw_answer = out.get("answer", "")
    source_documents = out.get("source_documents", []) or []
    grounded_fallback = bool(out.get("grounded_fallback", False))

    # Normalize LangChain Document dicts to the format verify/inject expect:
    # each source_for_verify entry should have at least: text, title/doc_id, chunk_id, page
    sources_for_verify = []
    for sd in source_documents:
        # sd is expected: {"page_content": ..., "metadata": {...}}
        if not isinstance(sd, dict):
            # best-effort cast
            try:
                page_content = getattr(sd, "page_content", None) or str(sd)
                metadata = getattr(sd, "metadata", None) or {}
            except Exception:
                page_content = str(sd)
                metadata = {}
        else:
            page_content = sd.get("page_content") or sd.get("text") or ""
            metadata = sd.get("metadata") or {}

        # Build the minimal chunk dict
        sources_for_verify.append({
            "text": page_content,
            "title": metadata.get("title") or metadata.get("doc_id") or metadata.get("doc_title"),
            "doc_id": metadata.get("doc_id"),
            "chunk_id": metadata.get("chunk_id"),
            "page": metadata.get("page"),
            "section": metadata.get("section"),
            "jurisdiction": metadata.get("jurisdiction"),
            "char_range": metadata.get("char_range"),
            "metadata": metadata  # keep raw metadata for frontends
        })

    # Inject inline citations using the verify + cite pipeline
    try:
        cited_answer, sentence_map = inject_citations(
            raw_answer,
            store,
            initial_chunks=sources_for_verify,
            threshold=threshold,
            top_k=verify_top_k
        )
    except Exception as e:
        # If injection fails, still return raw answer and sources with error info
        return jsonify({
            "cited_answer": raw_answer,
            "raw_answer": raw_answer,
            "sources": sources_for_verify,
            "verification": {"error": "citation injection failed", "details": str(e)},
            "grounded_fallback": grounded_fallback
        }), 200

    verify_report = {
        "total_sentences": len(sentence_map),
        "grounded_count": sum(1 for s in sentence_map if s.get("grounded")),
        "grounding_ratio": (sum(1 for s in sentence_map if s.get("grounded")) / len(sentence_map)) if sentence_map else 0.0,
        "sentences": sentence_map
    }

    response = {
        "cited_answer": cited_answer,
        "raw_answer": raw_answer,
        "sources": sources_for_verify,
        "verification": verify_report,
        "grounded_fallback": grounded_fallback
    }
    return jsonify(response)



if __name__ == '__main__':
    # FaissStore.load() is invoked in FaissStore.__init__, so index is available at startup if present.
    # Build chain_obj is done at import time; if you add many documents and want to reset LLM or chain,
    # you can rebuild by calling build_chain_from_store(store) and replacing chain_obj.
    app.run(host='0.0.0.0', port=7860, debug=True)
