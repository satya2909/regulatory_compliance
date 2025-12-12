import io
import pdfplumber
import re

def extract_text_from_file(content_bytes, filename):
    # Try pdfplumber for PDFs; otherwise attempt to decode as utf-8 text.
    if filename.lower().endswith('.pdf'):
        try:
            with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
                pages = [p.extract_text() or '' for p in pdf.pages]
                text = '\n'.join(pages)
                return text
        except Exception as e:
            print('pdfplumber failed:', e)
            try:
                return content_bytes.decode('utf-8', errors='ignore')
            except:
                return ''
    else:
        try:
            return content_bytes.decode('utf-8', errors='ignore')
        except:
            return ''

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
