# version_compare.py
"""
Version comparison engine.

Functions:
- compare_documents(old_doc, new_doc, model_name, match_threshold, modify_threshold)
    -> returns a dict with lists: added, removed, modified, unchanged

- compare_texts(old_text, new_text, ...)  # convenience wrapper that chunks texts then compares

Notes:
- Uses sentence-transformers embedding model (default: all-MiniLM-L6-v2).
- Uses cosine similarity to match chunks.
- Uses difflib to produce a small textual diff for modified chunks.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from typing import List, Dict, Any, Tuple, Optional
from utils import chunk_text

# Default embedding model (same as other modules)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

def _embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)

def _text_diff(a: str, b: str, n_chars: int = 300) -> str:
    # produce a small unified diff (line-oriented). Truncate long texts first.
    a_short = a if len(a) <= n_chars else a[:n_chars] + "..."
    b_short = b if len(b) <= n_chars else b[:n_chars] + "..."
    a_lines = a_short.splitlines() or [a_short]
    b_lines = b_short.splitlines() or [b_short]
    diff_lines = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
    return "\n".join(diff_lines)

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    # scores may be inner product or cosine depending on embedding handling.
    # Here we assume cosine_similarity was used and returned values in [-1,1].
    return scores

def compare_documents(
    old_doc: Dict[str, Any],
    new_doc: Dict[str, Any],
    model_name: str = _DEFAULT_MODEL,
    match_threshold: float = 0.72,
    modify_threshold: float = 0.90,
    top_k_for_matching: int = 1
) -> Dict[str, Any]:
    """
    Compare two documents (document dicts from your store).
    Each doc dict is expected to have keys: 'doc_id', 'title', 'chunks' where chunks is a list of dicts with 'chunk_id' and 'text', and optional metadata (page, char_range, section).

    Args:
      old_doc, new_doc: document dicts (from store.get_document(doc_id))
      model_name: sentence-transformers model name
      match_threshold: similarity threshold to consider a pair as a match (>= => matched)
      modify_threshold: similarity >= modify_threshold is considered unchanged; if match_threshold <= sim < modify_threshold => modified
      top_k_for_matching: how many top matches to consider while matching (keeps simple 1-to-1 mapping)

    Returns:
      {
        "old_doc_id": ...,
        "new_doc_id": ...,
        "added": [ {chunk metadata...} ],
        "removed": [ {chunk metadata...} ],
        "modified": [ { "old_chunk":..., "new_chunk":..., "score":..., "diff": "..." } ],
        "unchanged": [ { "old_chunk":..., "new_chunk":..., "score":... } ],
        "summary": {...}
      }
    """
    model = SentenceTransformer(model_name)

    old_chunks = old_doc.get("chunks", [])
    new_chunks = new_doc.get("chunks", [])

    old_texts = [c.get("text","") for c in old_chunks]
    new_texts = [c.get("text","") for c in new_chunks]

    # compute embeddings
    old_emb = _embed_texts(model, old_texts)  # shape (n_old, dim)
    new_emb = _embed_texts(model, new_texts)  # shape (n_new, dim)

    # If either side is empty, classify all as added/removed
    if old_emb.shape[0] == 0 and new_emb.shape[0] == 0:
        return {
            "old_doc_id": old_doc.get("doc_id"),
            "new_doc_id": new_doc.get("doc_id"),
            "added": [],
            "removed": [],
            "modified": [],
            "unchanged": [],
            "summary": {"n_old": 0, "n_new": 0}
        }
    if old_emb.shape[0] == 0:
        # all new chunks are added
        added = [dict(new_chunks[i], score=1.0) for i in range(len(new_chunks))]
        return {
            "old_doc_id": old_doc.get("doc_id"),
            "new_doc_id": new_doc.get("doc_id"),
            "added": added,
            "removed": [],
            "modified": [],
            "unchanged": [],
            "summary": {"n_old": 0, "n_new": len(new_chunks)}
        }
    if new_emb.shape[0] == 0:
        removed = [dict(old_chunks[i], score=1.0) for i in range(len(old_chunks))]
        return {
            "old_doc_id": old_doc.get("doc_id"),
            "new_doc_id": new_doc.get("doc_id"),
            "added": [],
            "removed": removed,
            "modified": [],
            "unchanged": [],
            "summary": {"n_old": len(old_chunks), "n_new": 0}
        }

    # compute pairwise cosine similarities (new vs old)
    # cosine_similarity rows: new_emb x old_emb => shape (n_new, n_old)
    sims = cosine_similarity(new_emb, old_emb)  # new x old

    # For each new chunk, find best matching old chunk (argmax over columns)
    best_old_idx_for_new = np.argmax(sims, axis=1)  # len = n_new
    best_score_for_new = sims[np.arange(sims.shape[0]), best_old_idx_for_new]

    # We'll also want reverse mapping to detect old chunks unmatched
    # For each old chunk, best matching new chunk index and score
    best_new_idx_for_old = np.argmax(sims.T, axis=1)  # len = n_old
    best_score_for_old = sims.T[np.arange(sims.T.shape[0]), best_new_idx_for_old]

    n_old = len(old_chunks)
    n_new = len(new_chunks)

    matched_old = set()
    matched_new = set()

    added = []
    removed = []
    modified = []
    unchanged = []

    # First pass: match new->old using match_threshold
    for new_i in range(n_new):
        old_i = int(best_old_idx_for_new[new_i])
        score = float(best_score_for_new[new_i])
        # If score >= match_threshold, consider them matched (may be modified or unchanged)
        if score >= match_threshold:
            matched_new.add(new_i)
            matched_old.add(old_i)
            # classify modified vs unchanged using modify_threshold
            if score >= modify_threshold:
                unchanged.append({
                    "old_chunk": old_chunks[old_i],
                    "new_chunk": new_chunks[new_i],
                    "score": score
                })
            else:
                diff_text = _text_diff(old_chunks[old_i].get("text",""), new_chunks[new_i].get("text",""))
                modified.append({
                    "old_chunk": old_chunks[old_i],
                    "new_chunk": new_chunks[new_i],
                    "score": score,
                    "diff": diff_text
                })
        else:
            # no match for this new chunk -> added
            added.append(dict(new_chunks[new_i], score=score))

    # Any old chunks not matched are removed
    for old_i in range(n_old):
        if old_i not in matched_old:
            removed.append(dict(old_chunks[old_i], score=float(best_score_for_old[old_i])))

    # summary
    summary = {
        "n_old": n_old,
        "n_new": n_new,
        "n_added": len(added),
        "n_removed": len(removed),
        "n_modified": len(modified),
        "n_unchanged": len(unchanged)
    }

    return {
        "old_doc_id": old_doc.get("doc_id"),
        "new_doc_id": new_doc.get("doc_id"),
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
        "summary": summary
    }

def compare_texts(
    old_text: str,
    new_text: str,
    model_name: str = _DEFAULT_MODEL,
    chunker = chunk_text,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience wrapper: chunk both texts using chunker (defaults to utils.chunk_text),
    build temporary doc dicts and call compare_documents.
    """
    old_chunks = chunker(old_text)
    new_chunks = chunker(new_text)
    old_doc = {"doc_id": "old_temp", "title": "old_text", "chunks": []}
    new_doc = {"doc_id": "new_temp", "title": "new_text", "chunks": []}
    for c in old_chunks:
        # ensure chunk has chunk_id
        old_doc["chunks"].append({
            "chunk_id": c.get("chunk_id") or c.get("id") or None,
            "text": c.get("text",""),
            "page": c.get("page"),
            "char_range": c.get("char_range"),
            "section": c.get("section"),
        })
    for c in new_chunks:
        new_doc["chunks"].append({
            "chunk_id": c.get("chunk_id") or c.get("id") or None,
            "text": c.get("text",""),
            "page": c.get("page"),
            "char_range": c.get("char_range"),
            "section": c.get("section"),
        })
    return compare_documents(old_doc, new_doc, model_name=model_name, **kwargs)
