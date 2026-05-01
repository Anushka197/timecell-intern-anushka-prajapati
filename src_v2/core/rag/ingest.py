"""
core/rag/ingest.py — ChromaDB ingestion pipeline

Handles chunking, embedding, and storing SEC filing text into ChromaDB.
All embeddings are generated via the local Ollama nomic-embed-text model.
"""

import chromadb
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from src_v2.core.llm import embed_batch

CHROMA_DIR = Path("src_v2/data/chromadb")
CHUNK_SIZE = 1000        # characters
CHUNK_OVERLAP = 150      # characters


@dataclass
class FilingChunk:
    text: str
    ticker: str
    company_name: str
    filing_type: str     # "10-K" or "10-Q"
    filing_date: str     # ISO 8601: "2024-11-01"
    accession_number: str
    chunk_index: int


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client at CHROMA_DIR."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(ticker: str) -> chromadb.Collection:
    """
    Returns the ChromaDB collection for the given ticker.
    Collection name = ticker.upper().
    Creates it if it doesn't exist, with metadata={"hnsw:space": "cosine"}.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=ticker.upper(),
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits text into overlapping chunks of approximately chunk_size characters.
    Splits on sentence boundaries ('. ', '! ', '? ') where possible.
    Produces overlapping chunks: the last `overlap` chars of chunk N appear
    at the start of chunk N+1.

    Returns a list of non-empty strings.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk = text[start:]
            stripped = chunk.strip()
            if stripped:
                chunks.append(stripped)
            break

        # Try to split on a sentence boundary within the window
        window = text[start:end]
        best_split = -1
        for delimiter in (". ", "! ", "? "):
            pos = window.rfind(delimiter)
            if pos != -1:
                candidate = pos + len(delimiter)  # include the delimiter
                if candidate > best_split:
                    best_split = candidate

        if best_split > 0:
            chunk = window[:best_split]
            # Next chunk starts overlap chars before the end of this chunk
            next_start = start + best_split - overlap
        else:
            chunk = window  # no sentence boundary found; hard split
            next_start = start + chunk_size - overlap

        stripped = chunk.strip()
        if stripped:
            chunks.append(stripped)

        # Ensure we always advance to avoid infinite loops
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)

        start = next_start

    return chunks


def is_filing_ingested(ticker: str, accession_number: str) -> bool:
    """
    Checks ChromaDB collection for the ticker to see if any document
    has metadata.accession_number == accession_number.
    Returns True if already ingested.
    """
    try:
        collection = get_or_create_collection(ticker)
        results = collection.get(
            where={"accession_number": accession_number},
            limit=1,
        )
        return len(results["ids"]) > 0
    except Exception:
        return False


def ingest_filing(
    filing_path: Path,
    ticker: str,
    company_name: str,
    filing_type: str,
    filing_date: str,
    accession_number: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Reads filing text, chunks it, embeds via nomic-embed-text, stores in ChromaDB.
    Skips if already ingested (idempotent).
    Calls progress_callback(chunks_done, total_chunks) if provided.
    Returns number of chunks ingested (0 if skipped).
    Chunk ID format: {ticker}_{accession_number}_chunk_{chunk_index}
    """
    # Idempotency check — skip if already ingested
    if is_filing_ingested(ticker, accession_number):
        return 0

    # Read PDF text
    import pypdf

    reader = pypdf.PdfReader(str(filing_path))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text += page_text

    # Chunk the text
    chunks = chunk_text(full_text)
    if not chunks:
        return 0

    total = len(chunks)

    # Generate embeddings for all chunks
    embeddings = embed_batch(chunks)

    # Build ChromaDB document data
    ids = [f"{ticker}_{accession_number}_chunk_{i}" for i in range(total)]
    metadatas = [
        {
            "ticker": ticker,
            "company_name": company_name,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "accession_number": accession_number,
            "chunk_index": i,
        }
        for i in range(total)
    ]

    # Store in ChromaDB
    collection = get_or_create_collection(ticker)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    # Fire progress callbacks
    if progress_callback is not None:
        for i in range(total):
            progress_callback(i + 1, total)

    return total
