# verify.py
"""
Citation verification & grounding checker.

Usage:
    from verify import verify_answer
    report = verify_answer(answer_text, store, initial_chunks=sources, threshold=0.65)
    # report contains sentence-level grounding info and overall stats.

Notes:
- `store` should implement `search(query, top_k)` returning a list of dicts with at least keys:
    - "score" (float): similarity score (higher = more similar)
    - "chunk_id" or "doc_id" (identifiers)
    - "text" (the chunk text)
  All current stores in the skeleton (in-memory, persistent, FAISS) implement this.
- `initial_chunks` is optional: pass the top-k chunks already retrieved for the question (faster).
- Adjust `threshold` depending on your embedding/search scale (e.g., 0.6–0.75 is a reasonable start for cosine similarity).
"""

import re
from typing import List, Dict, Optional

# Simple sentence splitter (lightweight, avoids extra deps)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    # Clean up whitespace and split
    if not text:
        return []
    text = text.strip()
    # Protect against very long outputs: split on newlines first
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        splits = _SENTENCE_SPLIT_RE.split(line)
        for s in splits:
            s = s.strip()
            if s:
                parts.append(s)
    return parts

def _best_support_from_chunks(sentence: str, chunks: List[Dict], top_k: int = 1) -> Optional[Dict]:
    """
    Find best supporting chunk among the provided `chunks`.
    `chunks` is a list of dicts with at least 'text' and optionally 'score', 'doc_id', 'chunk_id'.
    Returns the best-support dict with keys: {'chunk': chunk, 'score': score, 'match_type': 'substring'|'score'} or None.
    """
    if not chunks:
        return None

    # 1) substring checks (strong support)
    for ch in chunks:
        txt = (ch.get("text") or "").strip()
        if not txt:
            continue
        # if the sentence is fully contained in the chunk, it's strong support
        if sentence in txt or txt in sentence:
            # if chunk has a score, prefer it; else default to 1.0 for substring match
            score = float(ch.get("score", 1.0))
            return {"chunk": ch, "score": score, "match_type": "substring"}

    # 2) fallback to existing score if present: pick top chunk by score
    best = None
    best_score = None
    for ch in chunks:
        sc = ch.get("score")
        if sc is None:
            continue
        try:
            scf = float(sc)
        except Exception:
            continue
        if best is None or scf > best_score:
            best = ch
            best_score = scf
    if best is not None:
        return {"chunk": best, "score": float(best_score), "match_type": "score"}

    return None

def verify_answer(
    answer: str,
    store,
    initial_chunks: Optional[List[Dict]] = None,
    threshold: float = 0.65,
    top_k: int = 1,
    sentence_min_length: int = 10
) -> Dict:
    """
    Verify each sentence in `answer` for grounding.

    Parameters:
    - answer: LLM-generated text (string).
    - store: your store instance; must implement `search(query, top_k)` -> list[dict].
    - initial_chunks: optional list of chunks already retrieved for the question (prefer checking these first).
    - threshold: similarity threshold above which a match is considered grounding.
                 (Adjust per your embedding/search scale; default 0.65).
    - top_k: how many top results to request from store.search when falling back to global search.
    - sentence_min_length: ignore very short fragments.

    Returns a dict:
    {
      "total_sentences": N,
      "grounded_count": G,
      "grounding_ratio": G/N,
      "sentences": [
         {
            "sentence": "...",
            "grounded": True/False,
            "support": [ { "chunk": {...}, "score": 0.72, "match_type": "score" }, ... ]  # list, best first
         }, ...
      ]
    }
    """
    sentences = split_sentences(answer)
    results = []
    grounded_count = 0

    # Normalize initial_chunks to list
    if initial_chunks is None:
        initial_chunks = []

    for sent in sentences:
        sent_stripped = sent.strip()
        if len(sent_stripped) < sentence_min_length:
            # treat short fragments conservatively as ungrounded but include them
            results.append({"sentence": sent_stripped, "grounded": False, "support": []})
            continue

        # 1) Check initial_chunks (fast)
        support = []
        best = _best_support_from_chunks(sent_stripped, initial_chunks, top_k=top_k)
        if best:
            support.append(best)
        else:
            # 2) Ask the store for the best match (global search)
            if not hasattr(store, "search"):
                # cannot do a global search, so we mark ungrounded
                results.append({"sentence": sent_stripped, "grounded": False, "support": []})
                continue

            try:
                search_hits = store.search(sent_stripped, top_k=top_k)
            except Exception:
                # if store.search fails, mark ungrounded
                results.append({"sentence": sent_stripped, "grounded": False, "support": []})
                continue

            # pick best support from returned hits
            best = _best_support_from_chunks(sent_stripped, search_hits, top_k=top_k)
            if best:
                support.append(best)

        # Decide if grounded: any support with score >= threshold OR substring match
        is_grounded = False
        filtered_support = []
        for s in support:
            mt = s.get("match_type")
            sc = s.get("score", 0.0)
            if mt == "substring":
                is_grounded = True
            else:
                try:
                    # if store is FAISS (IndexFlatIP normalized), scores are inner-product in [-1,1]
                    # We assume higher is better. Threshold may need tuning per store.
                    if sc >= threshold:
                        is_grounded = True
                except Exception:
                    pass
            filtered_support.append(s)

        if is_grounded:
            grounded_count += 1

        results.append({"sentence": sent_stripped, "grounded": is_grounded, "support": filtered_support})

    total = len(results)
    ratio = (grounded_count / total) if total > 0 else 0.0

    return {
        "total_sentences": total,
        "grounded_count": grounded_count,
        "grounding_ratio": ratio,
        "sentences": results
    }
