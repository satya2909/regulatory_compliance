# cite.py (enhanced citation formatting with section & char_range)
"""
Inline citation injector with richer metadata support.

Key improvements:
- _format_citation_from_chunk now tries to include:
    title (or short doc id) | chunk_id | section (if present) | jurisdiction (if present) | p:page | char:char_start-char_end
- Works with chunk dicts coming from:
    - store.search outputs (keys like 'text','doc_id','chunk_id','title','page','char_range','section','jurisdiction')
    - LangChain Document metadata (under 'metadata')
    - verify.py support dicts where support['chunk'] contains the chunk dict

Usage and behavior are identical to the previous `cite.py`:
    cited_answer, sentence_map = inject_citations(answer, store, initial_chunks=sources, threshold=0.65)

Note: to show section/char_range you must ensure your chunker (utils.chunk_text) includes those fields in chunk metadata.
"""

from typing import List, Tuple, Optional, Dict, Any
from verify import verify_answer, split_sentences  # reuse existing verifier utilities

def _short_id(x: str, length: int = 12) -> str:
    if not x:
        return ""
    s = str(x)
    return s if len(s) <= length else s[:length]

def _format_citation_from_chunk(chunk: Dict[str, Any]) -> str:
    """
    Build a compact citation string from a chunk dict or a LangChain Document metadata dict.
    Prefer readable fields in this order:
      title -> doc_title -> metadata.title -> doc_id
    Then append chunk_id (short), section, jurisdiction, page, and char_range if available.

    Examples:
      "RBI-AML-2023 | chnk1234 | Sec 4.2 | India | p:12 | char:1024-1178"
      "doc_abc123 | 7f1a2b3c | p:5"
    """
    if not isinstance(chunk, dict):
        return "source"

    # If the chunk is wrapped (e.g., support item), try to unwrap
    # Many verify supports are like: {"chunk": {...}, "score":..., "match_type":...}
    if "chunk" in chunk and isinstance(chunk["chunk"], dict):
        core = chunk["chunk"]
    else:
        core = chunk

    # Some inputs might have a 'metadata' key (LangChain Document). Normalize it.
    meta = {}
    if core.get("metadata") and isinstance(core.get("metadata"), dict):
        meta.update(core.get("metadata"))

    # Merge top-level keys from core into meta if not present
    for k in ("title", "doc_title", "doc_id", "chunk_id", "page", "char_range", "section", "jurisdiction"):
        if k in core and k not in meta:
            meta[k] = core[k]

    # Determine left-most identifier (prefer title)
    title = meta.get("title") or meta.get("doc_title") or None
    if not title:
        # Try to derive a human-friendly title from doc_id if present
        doc_id = meta.get("doc_id")
        if doc_id:
            title = _short_id(doc_id, 12)
        else:
            # fallback to a generic label
            title = "doc"

    parts = [str(title)]

    # chunk id (short)
    chunk_id = meta.get("chunk_id") or meta.get("id")
    if chunk_id:
        parts.append(_short_id(chunk_id, 8))

    # section (if present)
    section = meta.get("section")
    if section:
        # keep section short-ish
        parts.append(f"sec:{str(section)}")

    # jurisdiction (country/authority)
    jurisdiction = meta.get("jurisdiction") or meta.get("juris")
    if jurisdiction:
        parts.append(str(jurisdiction))

    # page number
    page = meta.get("page")
    if page is not None:
        parts.append(f"p:{page}")

    # char_range: can be tuple/list [start,end] or "start-end" string
    char_range = meta.get("char_range")
    if char_range:
        try:
            if isinstance(char_range, (list, tuple)) and len(char_range) >= 2:
                start, end = int(char_range[0]), int(char_range[1])
                parts.append(f"char:{start}-{end}")
            elif isinstance(char_range, str) and "-" in char_range:
                parts.append(f"char:{char_range}")
            else:
                # single number
                parts.append(f"char:{char_range}")
        except Exception:
            # ignore parsing errors, but still include raw value
            parts.append(f"char:{char_range}")

    # Join with " | "
    cite_str = " | ".join(parts)
    return cite_str

def inject_citations(
    answer: str,
    store,
    initial_chunks: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.65,
    top_k: int = 1,
    sentence_min_length: int = 10,
    citation_template: str = "[{cite}]"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Inject inline citations into `answer` using verification supports.

    Returns:
    - cited_answer: str
    - sentence_map: list of per-sentence dicts:
        {
          "sentence": "...",
          "grounded": True/False,
          "citation": "[Title | chunkid | sec:... | p:...]" or None,
          "supports": [ { "chunk": {...}, "score": 0.72, "match_type": "score" }, ... ]
        }
    """
    report = verify_answer(answer, store, initial_chunks=initial_chunks, threshold=threshold, top_k=top_k, sentence_min_length=sentence_min_length)
    sentence_entries = report.get("sentences", [])

    cited_sentences = []
    sentence_map = []

    for entry in sentence_entries:
        sent = entry.get("sentence", "").strip()
        grounded = bool(entry.get("grounded", False))
        supports = entry.get("support", []) or []
        citation_text = None

        if grounded and supports:
            # choose best support (the first)
            best = supports[0] if isinstance(supports, list) and supports else None
            if isinstance(best, dict):
                # unwrap if necessary
                chunk = best.get("chunk") if "chunk" in best and isinstance(best.get("chunk"), dict) else best.get("chunk") or best
            else:
                chunk = None

            # if we don't have chunk as dict, maybe initial_chunks contain a plain mapping
            if chunk is None and isinstance(supports[0], dict):
                # fallback attempt
                chunk = supports[0].get("chunk") or supports[0]

            if isinstance(chunk, dict):
                core_cite = _format_citation_from_chunk(chunk)
                citation_text = citation_template.format(cite=core_cite)
            else:
                citation_text = citation_template.format(cite="source")

        # Append citation to sentence (after punctuation) or at end
        if citation_text:
            if sent.endswith((".", "!", "?", ";", ":")):
                cited = f"{sent} {citation_text}"
            else:
                cited = f"{sent}. {citation_text}"
        else:
            cited = sent

        cited_sentences.append(cited)
        sentence_map.append({
            "sentence": sent,
            "grounded": grounded,
            "citation": citation_text,
            "supports": supports
        })

    cited_answer = "  ".join(cited_sentences).strip()
    return cited_answer, sentence_map
