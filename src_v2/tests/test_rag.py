"""
Tests for core/rag/ingest.py — ChromaDB ingestion pipeline.

Task 8.1: Property 8 — Filing Ingestion Idempotence
  Validates: Requirements 10.4, 9.5

Task 8.2: Property 10 — ChromaDB Chunk Metadata Completeness
  Validates: Requirements 10.2, 10.3

Task 8.3: Unit tests for ingestion pipeline
  Validates: Requirements 10.4
"""

import pathlib
import tempfile
from io import BytesIO
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src_v2.core.rag.ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_text,
    get_or_create_collection,
    ingest_filing,
    is_filing_ingested,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_chroma_client(tmp_path: pathlib.Path):
    """Return a ChromaDB PersistentClient backed by a temp directory."""
    import chromadb
    return chromadb.PersistentClient(path=str(tmp_path / "chromadb"))


def _patch_chroma(tmp_path: pathlib.Path):
    """
    Context manager that patches get_chroma_client() in ingest.py to use
    a temp directory, preventing pollution of the real database.
    """
    client = _make_chroma_client(tmp_path)
    return patch("src_v2.core.rag.ingest.get_chroma_client", return_value=client)


def _fake_pdf_bytes(text: str = "Hello world. This is a test filing.") -> bytes:
    """
    Build a minimal valid PDF in memory containing `text`.
    Uses pypdf's PdfWriter so we don't need an external file.
    """
    import pypdf

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    # pypdf blank pages have no text; we'll mock PdfReader instead
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _fake_embed_batch(texts: list[str]) -> list[list[float]]:
    """Return deterministic unit-length embeddings (dim=4) for testing."""
    return [[float(i % 4 == j) for j in range(4)] for i, _ in enumerate(texts)]


# ── Unit tests: chunk_text ─────────────────────────────────────────────────

def test_chunk_text_empty_text_returns_empty_list() -> None:
    """Empty string must return an empty list."""
    assert chunk_text("") == []


def test_chunk_text_single_chunk_for_short_text() -> None:
    """Text shorter than chunk_size must return exactly one chunk."""
    short = "Hello world. This is a short sentence."
    result = chunk_text(short, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    assert len(result) == 1
    assert result[0] == short.strip()


def test_chunk_text_produces_overlapping_chunks() -> None:
    """
    The last `overlap` characters of chunk N must appear at the start of chunk N+1.
    We use a text long enough to produce at least 2 chunks.
    """
    # Build a text that is definitely longer than one chunk
    # Use simple repeated sentences so there are sentence boundaries
    sentence = "This is a sentence. "
    text = sentence * 200  # ~4000 chars, well above CHUNK_SIZE=1000

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    assert len(chunks) >= 2, "Expected at least 2 chunks for long text"

    for i in range(len(chunks) - 1):
        # The tail of chunk[i] should appear at the start of chunk[i+1]
        tail = chunks[i][-CHUNK_OVERLAP:]
        head = chunks[i + 1][:CHUNK_OVERLAP]
        # Allow for whitespace stripping differences at boundaries
        assert tail.strip() in chunks[i + 1] or head.strip() in chunks[i], (
            f"Overlap not found between chunk {i} and chunk {i+1}.\n"
            f"  tail of chunk[{i}]: {repr(tail)}\n"
            f"  head of chunk[{i+1}]: {repr(head)}"
        )


def test_chunk_text_no_empty_chunks() -> None:
    """chunk_text must never return empty strings."""
    text = "A" * 5000
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    for chunk in chunks:
        assert chunk.strip() != ""


def test_chunk_text_covers_all_content() -> None:
    """
    All content from the original text must appear across the chunks.
    We verify this by checking that the concatenation of all chunks (ignoring
    whitespace differences) contains all the unique words from the original text.
    """
    text = "The quick brown fox. Jumps over the lazy dog! Really? Yes indeed."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    # Join all chunks and normalise whitespace
    combined = " ".join(chunks)
    # Every word from the original should appear somewhere across all chunks
    # (overlap means words near boundaries appear in multiple chunks)
    original_words = set(w.strip(".,!?") for w in text.split())
    combined_words = set(w.strip(".,!?") for w in combined.split())
    missing = original_words - combined_words
    assert not missing, f"Words missing from chunks: {missing}"


# ── Unit tests: is_filing_ingested ─────────────────────────────────────────

def test_is_filing_ingested_returns_false_for_unknown(tmp_path: pathlib.Path) -> None:
    """is_filing_ingested must return False for an accession number not in the DB."""
    with _patch_chroma(tmp_path):
        result = is_filing_ingested("AAPL", "0000000000-00-000000")
    assert result is False


def test_is_filing_ingested_returns_true_after_ingestion(tmp_path: pathlib.Path) -> None:
    """
    After manually adding a document with accession_number metadata,
    is_filing_ingested must return True for that accession number.
    """
    accession = "0000320193-24-000123"
    ticker = "AAPL"

    with _patch_chroma(tmp_path) as mock_client:
        # Manually add a document to the collection
        collection = get_or_create_collection(ticker)
        collection.add(
            ids=[f"{ticker}_{accession}_chunk_0"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            documents=["Some filing text."],
            metadatas=[{
                "ticker": ticker,
                "company_name": "Apple Inc.",
                "filing_type": "10-K",
                "filing_date": "2024-11-01",
                "accession_number": accession,
                "chunk_index": 0,
            }],
        )
        result = is_filing_ingested(ticker, accession)

    assert result is True


# ── Unit tests: ingest_filing ──────────────────────────────────────────────

def test_ingest_filing_returns_zero_on_second_call(tmp_path: pathlib.Path) -> None:
    """
    Calling ingest_filing twice with the same accession number must return 0
    on the second call (idempotency).
    """
    ticker = "MSFT"
    accession = "0000789019-24-000001"
    filing_path = tmp_path / "filing.pdf"
    filing_path.write_bytes(b"dummy")  # content doesn't matter; we mock PdfReader

    filing_text = "Revenue increased significantly. Net income rose. " * 30  # ~1500 chars

    with _patch_chroma(tmp_path):
        with patch("src_v2.core.rag.ingest.embed_batch", side_effect=_fake_embed_batch):
            with patch("pypdf.PdfReader") as mock_reader_cls:
                # Set up mock PdfReader to return our fake text
                mock_page = MagicMock()
                mock_page.extract_text.return_value = filing_text
                mock_reader = MagicMock()
                mock_reader.pages = [mock_page]
                mock_reader_cls.return_value = mock_reader

                # First call — should ingest chunks
                count1 = ingest_filing(
                    filing_path=filing_path,
                    ticker=ticker,
                    company_name="Microsoft Corporation",
                    filing_type="10-K",
                    filing_date="2024-06-30",
                    accession_number=accession,
                )
                assert count1 > 0, "First ingest should return chunk count > 0"

                # Second call — should be skipped
                count2 = ingest_filing(
                    filing_path=filing_path,
                    ticker=ticker,
                    company_name="Microsoft Corporation",
                    filing_type="10-K",
                    filing_date="2024-06-30",
                    accession_number=accession,
                )
                assert count2 == 0, "Second ingest of same filing must return 0"


def test_ingest_filing_progress_callback(tmp_path: pathlib.Path) -> None:
    """progress_callback must be called once per chunk with (done, total)."""
    ticker = "TSLA"
    accession = "0001318605-24-000001"
    filing_path = tmp_path / "filing.pdf"
    filing_path.write_bytes(b"dummy")

    filing_text = "Tesla reported strong earnings. Revenue grew. " * 30

    calls: list[tuple[int, int]] = []

    def callback(done: int, total: int) -> None:
        calls.append((done, total))

    with _patch_chroma(tmp_path):
        with patch("src_v2.core.rag.ingest.embed_batch", side_effect=_fake_embed_batch):
            with patch("pypdf.PdfReader") as mock_reader_cls:
                mock_page = MagicMock()
                mock_page.extract_text.return_value = filing_text
                mock_reader = MagicMock()
                mock_reader.pages = [mock_page]
                mock_reader_cls.return_value = mock_reader

                count = ingest_filing(
                    filing_path=filing_path,
                    ticker=ticker,
                    company_name="Tesla Inc.",
                    filing_type="10-Q",
                    filing_date="2024-09-30",
                    accession_number=accession,
                    progress_callback=callback,
                )

    assert len(calls) == count
    for i, (done, total) in enumerate(calls):
        assert done == i + 1
        assert total == count


_ASCII_UPPER = st.text(
    min_size=3,
    max_size=6,
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
)

# ── Property 8: Filing Ingestion Idempotence ──────────────────────────────
# Validates: Requirements 10.4, 9.5

@given(
    ticker=_ASCII_UPPER,
    accession=st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    filing_text=st.text(min_size=100, max_size=3000),
)
@settings(max_examples=20)
def test_property_8_filing_ingestion_idempotence(
    ticker: str,
    accession: str,
    filing_text: str,
) -> None:
    """
    **Property 8: Filing Ingestion Idempotence**
    **Validates: Requirements 10.4, 9.5**

    Ingesting the same filing twice must not increase the chunk count in ChromaDB.
    The collection size after the second ingestion must equal the size after the first.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        filing_path = tmp_path / "filing.pdf"
        filing_path.write_bytes(b"dummy")

        with _patch_chroma(tmp_path):
            with patch("src_v2.core.rag.ingest.embed_batch", side_effect=_fake_embed_batch):
                with patch("pypdf.PdfReader") as mock_reader_cls:
                    mock_page = MagicMock()
                    mock_page.extract_text.return_value = filing_text
                    mock_reader = MagicMock()
                    mock_reader.pages = [mock_page]
                    mock_reader_cls.return_value = mock_reader

                    # First ingestion
                    ingest_filing(
                        filing_path=filing_path,
                        ticker=ticker,
                        company_name="Test Corp",
                        filing_type="10-K",
                        filing_date="2024-01-01",
                        accession_number=accession,
                    )
                    collection = get_or_create_collection(ticker)
                    count_after_first = collection.count()

                    # Second ingestion — must be a no-op
                    ingest_filing(
                        filing_path=filing_path,
                        ticker=ticker,
                        company_name="Test Corp",
                        filing_type="10-K",
                        filing_date="2024-01-01",
                        accession_number=accession,
                    )
                    count_after_second = collection.count()

    assert count_after_second == count_after_first, (
        f"Collection grew from {count_after_first} to {count_after_second} "
        f"after second ingestion of the same filing."
    )


# ── Property 10: ChromaDB Chunk Metadata Completeness ─────────────────────
# Validates: Requirements 10.2, 10.3

@given(
    ticker=_ASCII_UPPER,
    company_name=st.text(min_size=1, max_size=50),
    filing_type=st.sampled_from(["10-K", "10-Q"]),
    filing_date=st.just("2024-11-01"),
    accession=st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    filing_text=st.text(min_size=100, max_size=3000),
)
@settings(max_examples=20)
def test_property_10_chunk_metadata_completeness(
    ticker: str,
    company_name: str,
    filing_type: str,
    filing_date: str,
    accession: str,
    filing_text: str,
) -> None:
    """
    **Property 10: ChromaDB Chunk Metadata Completeness**
    **Validates: Requirements 10.2, 10.3**

    After ingesting any filing, every chunk stored in ChromaDB must have
    non-empty ticker, company_name, filing_type, filing_date, and accession_number
    metadata fields.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        filing_path = tmp_path / "filing.pdf"
        filing_path.write_bytes(b"dummy")

        with _patch_chroma(tmp_path):
            with patch("src_v2.core.rag.ingest.embed_batch", side_effect=_fake_embed_batch):
                with patch("pypdf.PdfReader") as mock_reader_cls:
                    mock_page = MagicMock()
                    mock_page.extract_text.return_value = filing_text
                    mock_reader = MagicMock()
                    mock_reader.pages = [mock_page]
                    mock_reader_cls.return_value = mock_reader

                    count = ingest_filing(
                        filing_path=filing_path,
                        ticker=ticker,
                        company_name=company_name,
                        filing_type=filing_type,
                        filing_date=filing_date,
                        accession_number=accession,
                    )

                    if count == 0:
                        # Empty text produces no chunks — nothing to verify
                        return

                    collection = get_or_create_collection(ticker)
                    results = collection.get(include=["metadatas"])

    required_fields = ["ticker", "company_name", "filing_type", "filing_date", "accession_number"]
    for i, metadata in enumerate(results["metadatas"]):
        for field in required_fields:
            assert field in metadata, f"Chunk {i} missing metadata field '{field}'"
            assert metadata[field] != "", f"Chunk {i} has empty metadata field '{field}'"
            assert metadata[field] is not None, f"Chunk {i} has None metadata field '{field}'"


# ── Task 9.2: Unit tests for RAG engine ───────────────────────────────────
# Validates: Requirements 11.5, 11.6

def _make_engine_chroma_client(tmp_path: pathlib.Path):
    """Return a ChromaDB PersistentClient backed by a temp directory."""
    import chromadb
    return chromadb.PersistentClient(path=str(tmp_path / "chromadb_engine"))


def _patch_engine_chroma(tmp_path: pathlib.Path):
    """
    Context manager that patches get_chroma_client() in engine.py to use
    a temp directory, preventing pollution of the real database.
    """
    client = _make_engine_chroma_client(tmp_path)
    return patch("src_v2.core.rag.engine.get_chroma_client", return_value=client)


def _seed_collection(client, ticker: str, filing_dates: list[str], chunks_per_date: int = 3):
    """
    Seed a ChromaDB collection with synthetic chunks for the given filing dates.
    Uses deterministic unit-length embeddings (dim=4).
    """
    import chromadb
    collection = client.get_or_create_collection(
        name=ticker.upper(),
        metadata={"hnsw:space": "cosine"},
    )
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    idx = 0
    for date in filing_dates:
        for chunk_i in range(chunks_per_date):
            chunk_id = f"{ticker}_{date}_chunk_{chunk_i}"
            ids.append(chunk_id)
            # Simple deterministic embedding
            embeddings.append([float(idx % 4 == j) for j in range(4)])
            documents.append(f"Filing text for {ticker} on {date}, chunk {chunk_i}.")
            metadatas.append({
                "ticker": ticker,
                "company_name": f"{ticker} Corp",
                "filing_type": "10-K",
                "filing_date": date,
                "accession_number": f"ACC-{date}-{chunk_i}",
                "chunk_index": chunk_i,
            })
            idx += 1
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return collection


def test_get_indexed_filing_dates_returns_sorted_order(tmp_path: pathlib.Path) -> None:
    """
    Seed a temp ChromaDB collection with chunks having different filing_dates;
    verify the returned list is sorted ascending.
    """
    from src_v2.core.rag.engine import get_indexed_filing_dates

    ticker = "AAPL"
    # Intentionally unsorted input dates
    dates = ["2024-11-01", "2023-05-15", "2024-02-28"]

    client = _make_engine_chroma_client(tmp_path)
    _seed_collection(client, ticker, dates)

    with patch("src_v2.core.rag.engine.get_chroma_client", return_value=client):
        result = get_indexed_filing_dates(ticker)

    assert result == sorted(dates), (
        f"Expected sorted dates {sorted(dates)}, got {result}"
    )


def test_what_changed_raises_value_error_with_fewer_than_2_filings(tmp_path: pathlib.Path) -> None:
    """
    Use a temp ChromaDB with only 1 filing date; verify what_changed raises ValueError.
    """
    from src_v2.core.rag.engine import what_changed

    ticker = "MSFT"
    client = _make_engine_chroma_client(tmp_path)
    _seed_collection(client, ticker, ["2024-11-01"])  # only 1 date

    with patch("src_v2.core.rag.engine.get_chroma_client", return_value=client):
        with pytest.raises(ValueError, match="At least two filings required"):
            what_changed(ticker)


def test_what_changed_raises_value_error_with_empty_collection(tmp_path: pathlib.Path) -> None:
    """
    Empty collection (no documents); verify what_changed raises ValueError.
    """
    from src_v2.core.rag.engine import what_changed

    ticker = "TSLA"
    client = _make_engine_chroma_client(tmp_path)
    # Create an empty collection (no documents)
    client.get_or_create_collection(
        name=ticker.upper(),
        metadata={"hnsw:space": "cosine"},
    )

    with patch("src_v2.core.rag.engine.get_chroma_client", return_value=client):
        with pytest.raises(ValueError, match="At least two filings required"):
            what_changed(ticker)
