"""
core/rag/engine.py — RAG Query Engine

Handles semantic retrieval from ChromaDB and LLM-powered Q&A over SEC filings.
All embedding and chat calls go through core/llm.py.
"""

import chromadb
from dataclasses import dataclass

from src_v2.core.llm import embed, chat
from src_v2.core.rag.ingest import get_chroma_client

TOP_K = 5


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    ticker: str
    filing_type: str
    filing_date: str
    accession_number: str
    similarity_score: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]


# ── Filing date index ──────────────────────────────────────────────────────

def get_indexed_filing_dates(ticker: str) -> list[str]:
    """
    Returns sorted list of unique filing_date values in the ticker's collection.
    Used to determine if 'What Changed?' is available.
    Returns empty list if collection doesn't exist or has no documents.
    """
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=ticker.upper())
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        dates = {
            m["filing_date"]
            for m in metadatas
            if m and "filing_date" in m
        }
        return sorted(dates)
    except Exception:
        return []


# ── Retrieval ──────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    ticker: str,
    top_k: int = TOP_K,
    filing_date_filter: str = None,
) -> list[RetrievedChunk]:
    """
    Embeds query, performs cosine similarity search in ticker's ChromaDB collection.
    Returns top_k most relevant chunks.
    similarity_score = 1 - distance
    If filing_date_filter is set, only retrieve chunks from that filing date.
    """
    embedding = embed(query)

    where_filter = None
    if filing_date_filter is not None:
        where_filter = {"filing_date": filing_date_filter}

    try:
        client = get_chroma_client()
        collection = client.get_collection(name=ticker.upper())
    except Exception:
        return []

    query_kwargs = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    chunks: list[RetrievedChunk] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                text=doc,
                ticker=meta.get("ticker", ticker),
                filing_type=meta.get("filing_type", ""),
                filing_date=meta.get("filing_date", ""),
                accession_number=meta.get("accession_number", ""),
                similarity_score=1.0 - dist,
            )
        )

    return chunks


# ── Full RAG pipeline ──────────────────────────────────────────────────────

def query(question: str, ticker: str) -> RAGResponse:
    """
    Full RAG pipeline: retrieve top_k chunks, build prompt, call LLM.
    Returns RAGResponse with answer and source citations.
    """
    chunks = retrieve(question, ticker, top_k=TOP_K)

    # Build numbered context from retrieved chunks
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"[{i}] ({chunk.filing_type}, {chunk.filing_date})\n{chunk.text}"
        )
    context = "\n\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst assistant. "
                "Answer questions based on the provided SEC filing excerpts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Here are relevant excerpts from SEC filings:\n\n"
                f"{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    response = chat(messages)
    return RAGResponse(answer=response, sources=chunks)


# ── What Changed ───────────────────────────────────────────────────────────

def what_changed(ticker: str) -> RAGResponse:
    """
    Retrieves chunks from the two most recent filings for the ticker.
    Prompts LLM to summarize changes in revenue, risk factors, and guidance.
    Returns RAGResponse with structured answer and sources from both filings.
    Raises ValueError if fewer than 2 filings are indexed for the ticker.
    """
    dates = get_indexed_filing_dates(ticker)
    if len(dates) < 2:
        raise ValueError("At least two filings required for comparison")

    # Two most recent dates (last two in ascending-sorted list)
    older_date = dates[-2]
    newer_date = dates[-1]

    older_chunks = retrieve(
        "revenue risk factors guidance outlook",
        ticker,
        top_k=TOP_K,
        filing_date_filter=older_date,
    )
    newer_chunks = retrieve(
        "revenue risk factors guidance outlook",
        ticker,
        top_k=TOP_K,
        filing_date_filter=newer_date,
    )

    all_sources = older_chunks + newer_chunks

    # Build context sections for each filing
    def _format_chunks(chunks: list[RetrievedChunk], label: str) -> str:
        lines = [f"=== {label} ==="]
        for i, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[{i}] ({chunk.filing_type}, {chunk.filing_date})\n{chunk.text}"
            )
        return "\n\n".join(lines)

    older_context = _format_chunks(older_chunks, f"Older Filing ({older_date})")
    newer_context = _format_chunks(newer_chunks, f"Newer Filing ({newer_date})")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst assistant. "
                "Answer questions based on the provided SEC filing excerpts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Compare the following two SEC filings for {ticker.upper()} "
                f"and summarize what changed between them.\n\n"
                f"{older_context}\n\n"
                f"{newer_context}\n\n"
                f"Please provide a structured summary with the following sections:\n"
                f"1. Revenue Changes\n"
                f"2. Risk Factor Changes\n"
                f"3. Guidance Changes\n\n"
                f"Be specific about what increased, decreased, or was newly added/removed."
            ),
        },
    ]

    response = chat(messages)
    return RAGResponse(answer=response, sources=all_sources)
