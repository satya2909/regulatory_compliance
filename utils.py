import io
import pdfplumber
import re
from docx import Document

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()

    if filename.endswith(".pdf"):
        text = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])

    elif filename.endswith(".doc"):
        raise ValueError(
            "Legacy .doc files are not supported. Please upload .docx or PDF."
        )

    else:
        raise ValueError("Unsupported file format")

def chunk_text(text, chunk_size=800, overlap=150):
    # Basic chunker: splits on paragraphs and then by token approximations (characters)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    for para in paragraphs:
        start = 0
        while start < len(para):
            end = start + chunk_size
            chunk_text = para[start:end]
            char_range = (start, min(end, len(para)))
            chunks.append({"text": chunk_text, "char_range": char_range})
            start = end - overlap
    # If no paragraphs found, fallback to sliding window on whole text
    if not chunks:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"text": text[start:end], "char_range": (start, min(end, len(text)))})
            start = end - overlap
    return chunks
