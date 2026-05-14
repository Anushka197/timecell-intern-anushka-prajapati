"""
ingest.py — Step 2: Chunking & Embedding
===========================================
Loads cached parsed documents, chunks them, and stores them in ChromaDB.
"""

import os
import json
import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("PortfolioIngestor")

load_dotenv()

CACHE_FILE = Path("./data/parsed_docs.json")
CHROMA_DIR = Path("./chroma_db")
CHROMA_COLLECTION = "apple_10k"

class PortfolioIngestor:
    def __init__(self):
        self._validate_env()
        self._configure_settings()
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        logger.info("ChromaDB client initialised at '%s'", CHROMA_DIR)

    def _validate_env(self):
        required = ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}.")

    def _configure_settings(self):
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=os.environ["OPENROUTER_API_KEY"],
            api_base=os.environ["OPENROUTER_BASE_URL"],
        )
        
        # Use the official OpenRouter class here
        Settings.llm = OpenRouter(
            model="openai/gpt-4o",
            api_key=os.environ["OPENROUTER_API_KEY"],
            # No need to specify api_base here; the class knows it's OpenRouter
        )
        logger.info("LLM configured via official OpenRouter integration.")

        
    def _load_cached_documents(self) -> list:
        if not CACHE_FILE.exists():
            raise FileNotFoundError(f"Cache file {CACHE_FILE} not found. Please run 1_parse.py first.")
        
        logger.info("Loading parsed documents from %s...", CACHE_FILE)
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            docs_dicts = json.load(f)
        
        # Convert raw dictionaries back into LlamaIndex Document objects
        documents = [Document.from_dict(d) for d in docs_dicts]
        logger.info("Loaded %d document sections from cache.", len(documents))
        return documents

    def _chunk_documents(self, documents: list) -> list:
        logger.info("Stage 1: Parsing Markdown structure…")
        
        # Replace the buggy Element parser with the standard Markdown parser
        md_parser = MarkdownNodeParser()
        
        md_nodes = md_parser.get_nodes_from_documents(documents)
        logger.info("  → %d nodes after Markdown parsing.", len(md_nodes))

        logger.info("Stage 2: Sentence splitting…")
        sentence_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            include_metadata=True,
        )
        final_nodes = sentence_splitter.get_nodes_from_documents(md_nodes)
        logger.info("  → %d final nodes ready for embedding.", len(final_nodes))
        return final_nodes

    def load_index(self) -> VectorStoreIndex:
        collection = self.chroma_client.get_or_create_collection(CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if collection.count() > 0:
            logger.info("Existing collection found. Loading index from ChromaDB…")
            return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        logger.info("No existing collection found. Starting ingestion pipeline…")
        documents = self._load_cached_documents()
        nodes = self._chunk_documents(documents)

        logger.info("Embedding and persisting %d nodes to ChromaDB…", len(nodes))
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
        logger.info("✓ Ingestion complete. Index persisted.")
        return index
    
    def get_indexed_files(self) -> list[dict]:
        """
        Return a list of filing metadata dicts for files that have been indexed.
        Used by the Streamlit sidebar to display index status.
        """
        filing_metadata = {
            "2020.pdf": {"year": 2020},
            "2021.pdf": {"year": 2021},
            "2022.pdf": {"year": 2022},
            "2023.pdf": {"year": 2023},
            "2024.pdf": {"year": 2024},
            "2025.pdf": {"year": 2025},
        }
        
        try:
            collection = self.chroma_client.get_collection(CHROMA_COLLECTION)
            # Sample metadata to find which fiscal years are present
            results = collection.get(include=["metadatas"], limit=10000)
            years_found: set[str] = set()
            
            for meta in results.get("metadatas", []):
                if meta and "fiscal_year" in meta:
                    years_found.add(str(meta["fiscal_year"]))

            status = []
            for filename, meta in filing_metadata.items():
                status.append(
                    {
                        "file": filename,
                        "year": meta["year"],
                        "indexed": str(meta["year"]) in years_found,
                    }
                )
            return status
        except Exception:
            return [
                {"file": f, "year": m["year"], "indexed": False}
                for f, m in filing_metadata.items()
            ]

if __name__ == "__main__":
    ingestor = PortfolioIngestor()
    index = ingestor.load_index()
    logger.info("Index is ready for querying!")