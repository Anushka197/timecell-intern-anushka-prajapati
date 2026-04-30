"""
parse_v3.py — Step 1: Structural Extraction (OCR)
===================================================
Uses LlamaParse to convert PDF tables to Markdown.
This acts as the "Data Intake" layer before passing to the local system.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from llama_parse import LlamaParse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("Parser_V3")

load_dotenv()

DATA_DIR = Path("./data")
CACHE_FILE = Path("./data/parsed_docs_v3.json") # V3 Path

FILING_METADATA = {
    "2020.pdf": {"year": 2020, "company": "Apple Inc.", "filing_type": "10-K"},
    "2021.pdf": {"year": 2021, "company": "Apple Inc.", "filing_type": "10-K"},
    "2022.pdf": {"year": 2022, "company": "Apple Inc.", "filing_type": "10-K"},
    "2023.pdf": {"year": 2023, "company": "Apple Inc.", "filing_type": "10-K"},
    "2024.pdf": {"year": 2024, "company": "Apple Inc.", "filing_type": "10-K"},
    "2025.pdf": {"year": 2025, "company": "Apple Inc.", "filing_type": "10-K"},
}

def parse_and_save():
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        raise EnvironmentError("Missing LLAMA_CLOUD_API_KEY.")

    parser = LlamaParse(
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
        result_type="markdown",
        verbose=True,
        language="en",
        parsing_instruction=(
            "This document is an SEC 10-K financial filing. "
            "CRITICAL: Identify all financial tables (Income Statement, Balance Sheet, Cash Flow, etc.). "
            "Extract tables with 100% structural accuracy using Markdown pipe syntax. "
            "Immediately before every table, write a one-sentence summary of what the table represents, "
            "including the fiscal year it applies to, so the context is embedded directly with the table data."
        ),
    )

    all_documents = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found in '%s'.", DATA_DIR)
        return

    for pdf_path in pdf_files:
        filename = pdf_path.name
        meta = FILING_METADATA.get(filename, {"year": "unknown", "company": "Apple Inc.", "filing_type": "10-K"})

        logger.info("Parsing '%s' (FY%s)…", filename, meta["year"])
        docs = parser.load_data(str(pdf_path))

        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source_file": filename,
                "fiscal_year": str(meta["year"]),
                "company": meta["company"],
                "filing_type": meta["filing_type"],
                "page_number": i + 1,
            })
        all_documents.extend(docs)

    docs_dicts = [doc.to_dict() for doc in all_documents]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(docs_dicts, f, indent=2)
    logger.info("✓ Documents successfully saved to V3 cache.")

if __name__ == "__main__":
    parse_and_save()