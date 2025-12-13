# pdf_diff.py
"""
Convert version comparison output into PDF highlight instructions
for side-by-side PDF diff viewer.
"""

from typing import Dict, List, Any

COLOR_MAP = {
    "added": "green",
    "removed": "red",
    "modified": "yellow"
}

def build_pdf_diff_highlights(compare_result: Dict[str, Any]):
    """
    Input: output of compare_documents(...)
    Output:
    {
      "old": { page: [highlight...] },
      "new": { page: [highlight...] }
    }
    """
    highlights = {"old": {}, "new": {}}

    # Removed → highlight only in OLD PDF
    for c in compare_result.get("removed", []):
        page = c.get("page")
        if page is None:
            continue
        highlights["old"].setdefault(page, []).append({
            "chunk_id": c.get("chunk_id"),
            "char_range": c.get("char_range"),
            "color": COLOR_MAP["removed"],
            "type": "removed",
            "text": c.get("text", "")[:300]
        })

    # Added → highlight only in NEW PDF
    for c in compare_result.get("added", []):
        page = c.get("page")
        if page is None:
            continue
        highlights["new"].setdefault(page, []).append({
            "chunk_id": c.get("chunk_id"),
            "char_range": c.get("char_range"),
            "color": COLOR_MAP["added"],
            "type": "added",
            "text": c.get("text", "")[:300]
        })

    # Modified → highlight in BOTH PDFs
    for c in compare_result.get("modified", []):
        old_c = c.get("old_chunk", {})
        new_c = c.get("new_chunk", {})

        if old_c.get("page") is not None:
            highlights["old"].setdefault(old_c["page"], []).append({
                "chunk_id": old_c.get("chunk_id"),
                "char_range": old_c.get("char_range"),
                "color": COLOR_MAP["modified"],
                "type": "modified",
                "text": old_c.get("text", "")[:300]
            })

        if new_c.get("page") is not None:
            highlights["new"].setdefault(new_c["page"], []).append({
                "chunk_id": new_c.get("chunk_id"),
                "char_range": new_c.get("char_range"),
                "color": COLOR_MAP["modified"],
                "type": "modified",
                "text": new_c.get("text", "")[:300]
            })

    return highlights
