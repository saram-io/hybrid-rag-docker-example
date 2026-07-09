import io
import pypdf

def extract_text(file_content: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        reader = pypdf.PdfReader(io.BytesIO(file_content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(file_content))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return file_content.decode("utf-8", errors="replace")

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " ", ""]
    return _split(text, separators, chunk_size, overlap)

def _split(text, separators, chunk_size, overlap):
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = next((s for s in separators if s in text), separators[-1])
    parts = text.split(sep)
    chunks, current = [], ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size and len(separators) > 1:
                chunks.extend(_split(part, separators[1:], chunk_size, overlap))
                current = ""
            else:
                current = part

    if current and current.strip():
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(chunks[i-1][-overlap:] + " " + chunks[i])
        chunks = overlapped

    return [c.strip() for c in chunks if c.strip()]
