# checklist.py
"""
Checklist generator for regulatory clauses.

Usage:
    from checklist import generate_checklist
    out = generate_checklist(store, "Generate AML onboarding checklist for digital accounts", top_k=6)
    # out is dict: { "success": True, "checklist": [...], "raw": "<llm output>", "sources": [...] }

Dependencies:
- Uses rag.LLMClient to call configured LLM provider (OpenAI/Gemini/Claude) if available.
- Falls back to returning grounded concatenated context if no LLM configured.

Notes:
- The LLM is instructed to return a strict JSON array of checklist items. Each item fields:
    {
      "id": "chk-1",
      "task": "Describe the task to perform",
      "rationale": "Why this task is required (brief), with citation like [Title | chunk_id | p:12]",
      "priority": "High|Medium|Low",
      "owner": "Compliance|KYC Team|Ops (suggested)",
      "evidence": [ { "doc_id": "...", "chunk_id": "...", "title":"...", "page": 12, "snippet": "..." }, ... ]
    }
- If LLM does not produce valid JSON, the module attempts to locate a JSON block in output; otherwise returns raw text in `raw`.
"""

import json
from typing import List, Dict, Any, Optional
from rag import LLMClient, build_prompt_from_chunks  # uses the rag.py helper we added earlier

def _build_checklist_prompt(chunks: List[Dict[str, Any]], user_request: str, max_items: int = 20) -> str:
    """
    Build a strict prompt asking the LLM to produce a JSON checklist using only the provided chunks.
    """
    context_parts = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("title") or c.get("doc_id") or c.get("metadata", {}).get("title") or f"doc_{i}"
        chunk_id = c.get("chunk_id") or c.get("metadata", {}).get("chunk_id") or str(i)
        page = c.get("page") or c.get("metadata", {}).get("page")
        text = c.get("text") or c.get("page_content") or c.get("metadata", {}).get("text","")
        snippet = (text[:600] + "...") if text and len(text) > 600 else (text or "")
        meta = {"title": title, "chunk_id": chunk_id, "page": page}
        context_parts.append(f"[{i}] {title} | {chunk_id} | p:{page if page is not None else '-'}\n{snippet}\n")

    context = "\n---\n".join(context_parts)

    prompt = f"""
You are an expert regulatory compliance assistant. Use ONLY the CONTEXT below to create an actionable compliance checklist in JSON format.
Do NOT invent facts beyond the context. If the context does not contain required information, say that the specific requirement is not present.

Context:
{context}

User request:
{user_request}

Instructions (VERY IMPORTANT) — must be followed strictly:
1) Return ONLY valid JSON. The top-level value MUST be an array of checklist items (no extra prose).
2) Each checklist item must include these keys:
   - id: string (unique short id, e.g., 'chk-1')
   - task: string (actionable task, imperative)
   - rationale: string (one-sentence rationale citing the context, include citation like [Title | chunk_id | p:12])
   - priority: one of [High, Medium, Low]
   - owner: suggested owner like 'Compliance', 'KYC Team', 'Ops'
   - evidence: array of evidence objects, each evidence object should include at least: doc_id or title, chunk_id, page, snippet (short)
3) Provide at most {max_items} items.
4) If you must paraphrase, match the meaning of the clause exactly and include the supporting chunk citation(s).
5) For items you cannot find any supporting text, include the item with rationale set to 'No supporting text found in provided documents' and evidence empty.

Produce output example:
[
  {{
    "id":"chk-1",
    "task":"Perform KYC for all new customers before account activation",
    "rationale":"RBI guidelines require identity verification for new customers [RBI AML 2023 | chnk1234 | p:12]",
    "priority":"High",
    "owner":"KYC Team",
    "evidence":[{{"title":"RBI AML 2023","chunk_id":"chnk1234","page":12,"snippet":"...text..."}}]
  }},
  ...
]

Now produce the checklist JSON array.
"""
    return prompt

def _extract_json_from_text(text: str) -> Optional[Any]:
    """
    Attempt to extract the first JSON structure from the model output.
    Returns parsed object or None.
    """
    if not text:
        return None
    text = text.strip()
    # Try direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON block between first '[' and last ']' or between '{' and matching '}'
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = text[first_bracket:last_bracket+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try to find first '{' ... '}' block (less likely since top-level must be array)
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None

def generate_checklist(
    store,
    user_request: str,
    top_k: int = 8,
    max_items: int = 20,
    llm_client: Optional[LLMClient] = None,
    return_raw_if_parse_fail: bool = True
) -> Dict[str, Any]:
    """
    Main entrypoint.

    Args:
      store: your FaissStore or DocStore exposing search(query, top_k)
      user_request: e.g., "Generate the AML onboarding checklist for digital accounts"
      top_k: how many supporting chunks to retrieve
      max_items: maximum items for the checklist
      llm_client: optional pre-configured LLMClient (if None, a default LLMClient() will be created)
      return_raw_if_parse_fail: if True and JSON parse fails, return the raw LLM output in 'raw' key

    Returns:
      {
        "success": True/False,
        "checklist": [ ... ]  OR None,
        "raw": "<llm output>",
        "sources": [ list of chunks used ],
        "error": "<error message if any>"
      }
    """
    # 1) retrieve top_k supporting chunks using store.search()
    try:
        # Using the user_request itself for retrieval is a practical approach
        retrieved = store.search(user_request, top_k=top_k)
    except Exception as e:
        return {"success": False, "error": f"store.search failed: {e}", "checklist": None}

    # 2) Build prompt (rooted in retrieved context)
    prompt = _build_checklist_prompt(retrieved, user_request, max_items=max_items)

    # 3) Call LLM
    try:
        llm = llm_client or LLMClient()
        raw_out = llm.generate(prompt)
    except Exception as e:
        return {"success": False, "error": f"LLM call failed: {e}", "checklist": None}

    # 4) Try to parse JSON
    parsed = _extract_json_from_text(raw_out)
    if parsed is None:
        # parsing failed
        if return_raw_if_parse_fail:
            return {"success": False, "error": "Failed to parse JSON from LLM output", "raw": raw_out, "sources": retrieved}
        else:
            return {"success": False, "error": "Failed to parse JSON from LLM output", "raw": raw_out, "sources": retrieved}

    # 5) Normalize evidence entries: ensure fields exist and include snippet truncated
    def _norm_evidence(evd):
        # expected keys: title/doc_id, chunk_id, page, snippet
        if not isinstance(evd, dict):
            return {}
        return {
            "title": evd.get("title") or evd.get("doc_id") or evd.get("document"),
            "chunk_id": evd.get("chunk_id"),
            "page": evd.get("page"),
            "snippet": (evd.get("snippet") or "")[:800]  # truncate snippet
        }

    checklist = []
    idx = 0
    for item in parsed:
        idx += 1
        # ensure basic fields
        item_id = item.get("id") or f"chk-{idx}"
        task = item.get("task") or item.get("action") or ""
        rationale = item.get("rationale") or ""
        priority = item.get("priority") or "Medium"
        owner = item.get("owner") or "Compliance"
        evidence_raw = item.get("evidence") or []
        evidence = [_norm_evidence(e) for e in evidence_raw if e]
        checklist.append({
            "id": item_id,
            "task": task,
            "rationale": rationale,
            "priority": priority,
            "owner": owner,
            "evidence": evidence
        })

    return {"success": True, "checklist": checklist, "raw": raw_out, "sources": retrieved}
