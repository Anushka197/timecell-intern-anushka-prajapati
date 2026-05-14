"""
parse.py — Step 1: PDF to Markdown Parsing
=============================================
Parses PDFs using LlamaParse and saves the output locally to a JSON file.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from llama_parse import LlamaParse

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("Parser")

load_dotenv()

DATA_DIR = Path("./data")
CACHE_FILE = Path("./data/parsed_docs.json")

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
        system_prompt=(
            "This is an Apple Inc. Annual Report (10-K) filing. "
            "Extract all financial tables, footnotes, and narrative sections accurately. "
            "Preserve table formatting using Markdown pipe syntax."
        ),
    )

    all_documents = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found in '%s'.", DATA_DIR)
        return

    logger.info("Found %d PDF files to parse.", len(pdf_files))

    for pdf_path in pdf_files:
        filename = pdf_path.name
        meta = FILING_METADATA.get(filename, {"year": "unknown", "company": "Apple Inc.", "filing_type": "10-K"})

        logger.info("Parsing '%s' (FY%s)…", filename, meta["year"])
        docs = parser.load_data(str(pdf_path))

        # Inject metadata
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source_file": filename,
                "fiscal_year": str(meta["year"]),
                "company": meta["company"],
                "filing_type": meta["filing_type"],
                "page_number": i + 1,
            })
        all_documents.extend(docs)

    logger.info("Parsing complete. Total document sections: %d", len(all_documents))

    # Save to JSON
    logger.info("Saving parsed documents to %s...", CACHE_FILE)
    docs_dicts = [doc.to_dict() for doc in all_documents]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(docs_dicts, f, indent=2)
    logger.info("✓ Documents successfully saved. You can now run the ingestion script.")

if __name__ == "__main__":
    parse_and_save()