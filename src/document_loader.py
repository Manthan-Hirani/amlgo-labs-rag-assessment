"""
Document Loader Module
======================
Handles PDF text extraction, cleaning, and sentence-aware chunking
for the RAG pipeline. Includes document summary generation for
improved overview queries.
"""

import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file, returning a list of dicts
    with 'page' (1-indexed) and 'text' keys.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text
            })
    doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing artifacts, normalizing whitespace,
    and fixing common OCR/extraction issues.
    """
    # Remove page numbers (standalone numbers on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix broken words across lines (hyphenation)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Normalize whitespace within lines
    text = re.sub(r'[ \t]+', ' ', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def extract_section_headers(full_text: str) -> list[str]:
    """
    Extract numbered section headers from the document text.
    Looks for patterns like '1. Introduction', '2. About eBay', etc.
    """
    headers = []
    # Match numbered sections: "1. Title", "2.1 Title", etc.
    pattern = r'^(\d+\.(?:\d+\.?)*)\s+([A-Z][^\n]{2,80})$'
    for match in re.finditer(pattern, full_text, re.MULTILINE):
        section_num = match.group(1)
        section_title = match.group(2).strip()
        headers.append(f"Section {section_num} {section_title}")

    # Also look for bold/uppercase headers without numbers
    pattern2 = r'^([A-Z][A-Z\s&]{5,60})$'
    for match in re.finditer(pattern2, full_text, re.MULTILINE):
        header = match.group(1).strip()
        if header and header not in headers and len(header.split()) <= 8:
            headers.append(header)

    return headers


def generate_summary_chunk(pages: list[dict]) -> dict:
    """
    Generate a synthetic document summary chunk from the first few pages
    and extracted section headers. This chunk is designed to match
    broad overview queries like "What is this document about?"
    
    Returns:
        A chunk dict with the document summary
    """
    # Get the full document text for header extraction
    full_text = "\n\n".join(clean_text(p["text"]) for p in pages)

    # Extract section headers
    headers = extract_section_headers(full_text)

    # Get the first page content (usually has the title and intro)
    first_page_text = clean_text(pages[0]["text"]) if pages else ""

    # Extract the document title (first meaningful line)
    title_lines = [
        line for line in first_page_text.split('\n')
        if line.strip() and len(line.strip()) > 3
    ]
    doc_title = title_lines[0] if title_lines else "Document"

    # Build a comprehensive summary
    summary_parts = [
        f"DOCUMENT OVERVIEW: This document is titled \"{doc_title}\".",
        f"The document spans {len(pages)} pages and covers the following main topics and sections:",
    ]

    if headers:
        summary_parts.append("Main sections include: " + "; ".join(headers[:20]) + ".")

    # Include the introduction/first page content for context
    intro_text = first_page_text[:1500] if first_page_text else ""
    if intro_text:
        summary_parts.append(f"\nIntroduction excerpt:\n{intro_text}")

    summary_text = "\n".join(summary_parts)

    return {
        "chunk_id": 0,
        "text": summary_text,
        "source_page": 0,  # 0 indicates synthetic/overview chunk
        "word_count": len(summary_text.split()),
        "is_summary": True,
    }


def chunk_documents(
    pages: list[dict],
    chunk_size: int = 1500,
    chunk_overlap: int = 300
) -> list[dict]:
    """
    Split document pages into overlapping chunks of 150-300 words
    using sentence-aware splitting with larger chunk sizes for
    better cross-section reasoning.
    
    Args:
        pages: List of dicts with 'page' and 'text' keys
        chunk_size: Target chunk size in characters (~300 words at ~5 chars/word)
        chunk_overlap: Overlap between consecutive chunks in characters
    
    Returns:
        List of chunk dicts with 'chunk_id', 'text', 'source_page', 'word_count'
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    # Start with the summary chunk at position 0
    summary_chunk = generate_summary_chunk(pages)
    chunks = [summary_chunk]
    chunk_id = 1  # Start regular chunks from ID 1

    for page_info in pages:
        cleaned = clean_text(page_info["text"])
        if not cleaned:
            continue

        page_chunks = splitter.split_text(cleaned)

        for chunk_text in page_chunks:
            word_count = len(chunk_text.split())
            if word_count < 5:  # Lower threshold to keep more content
                continue

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_page": page_info["page"],
                "word_count": word_count,
            })
            chunk_id += 1

    return chunks


def save_chunks(chunks: list[dict], output_dir: str) -> str:
    """
    Save processed chunks to a JSON file in the output directory.
    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return output_path


def load_chunks(chunks_dir: str) -> list[dict]:
    """Load previously saved chunks from JSON."""
    chunks_path = os.path.join(chunks_dir, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_document(pdf_path: str, chunks_dir: str = "chunks") -> list[dict]:
    """
    End-to-end document processing pipeline:
    1. Extract text from PDF
    2. Clean the text
    3. Generate document summary chunk
    4. Chunk into larger segments (1500 chars, ~300 words)
    5. Save to disk
    
    Returns the list of chunks.
    """
    print(f"[DOC] Extracting text from: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"   Found {len(pages)} pages with text")

    print("[CHUNK] Chunking documents (larger segments for better context)...")
    chunks = chunk_documents(pages)
    print(f"   Created {len(chunks)} chunks (including 1 summary chunk)")

    # Print chunk statistics (excluding summary chunk)
    regular_chunks = [c for c in chunks if not c.get("is_summary")]
    word_counts = [c["word_count"] for c in regular_chunks]
    print(f"   Regular chunk word range: {min(word_counts)}-{max(word_counts)}")
    print(f"   Average words per chunk: {sum(word_counts) / len(word_counts):.0f}")
    print(f"   Summary chunk words: {chunks[0]['word_count']}")

    output_path = save_chunks(chunks, chunks_dir)
    print(f"[SAVE] Chunks saved to: {output_path}")

    return chunks


if __name__ == "__main__":
    # Run standalone for testing
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "data" / "AI Training Document.pdf"
    chunks_dir = project_root / "chunks"
    chunks = process_document(str(pdf_path), str(chunks_dir))
    print(f"\n[DONE] Processed {len(chunks)} chunks.")
