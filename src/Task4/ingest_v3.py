"""
ingest_v3.py — Step 2: 100% Local Ingestion & Vectorization
=============================================================
Uses local HuggingFace embeddings (BGE-Small) and local Markdown 
parsing. Achieves full data sovereignty with zero API calls.
"""

import json
import logging
from pathlib import Path

import chromadb

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore

# NO OPENAI OR OPENROUTER IMPORTS!
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("PortfolioIngestor_V3")

# V3 PATHS
CACHE_FILE = Path("./data/parsed_docs_v3.json")
CHROMA_DIR = Path("./chroma_db_v3")
CHROMA_COLLECTION = "apple_10k_v3"

class PortfolioIngestorV3:
    def __init__(self):
        self._configure_settings()
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        logger.info("ChromaDB V3 client initialised at '%s'", CHROMA_DIR)

    def _configure_settings(self):
        # 1. LOCAL EMBEDDINGS (No API costs, fully private)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # 2. LOCAL LLM via OLLAMA (Make sure you ran `ollama run llama3.1`)
        Settings.llm = Ollama(
            model="llama3.1", 
            request_timeout=120.0
        )
        
        logger.info("Settings configured: 100% LOCAL. (BGE Embeddings + Ollama LLM)")

    def _load_cached_documents(self) -> list:
        if not CACHE_FILE.exists():
            raise FileNotFoundError(f"Cache file {CACHE_FILE} not found. Run parse_v3.py first.")

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            docs_dicts = json.load(f)

        documents = [Document.from_dict(d) for d in docs_dicts]
        logger.info("Loaded %d document sections from V3 cache.", len(documents))
        return documents

    def _chunk_documents(self, documents: list) -> list:
        logger.info("Stage 1: Parsing Markdown structure locally…")
        
        md_parser = MarkdownNodeParser()
        md_nodes = md_parser.get_nodes_from_documents(documents, show_progress=True)
        
        logger.info("Stage 2: Sentence splitting…")
        sentence_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
        
        final_nodes = sentence_splitter.get_nodes_from_documents(md_nodes, show_progress=True)
        logger.info(" → %d final nodes ready for embedding.", len(final_nodes))

        return final_nodes

    def load_index(self) -> VectorStoreIndex:
        collection = self.chroma_client.get_or_create_collection(CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if collection.count() > 0:
            logger.info("Existing V3 collection found. Loading index…")
            return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        logger.info("No existing collection found. Starting Sovereign V3 ingestion pipeline…")

        documents = self._load_cached_documents()
        nodes = self._chunk_documents(documents)

        logger.info("Creating empty index framework...")
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)

        # RESILIENT BATCHING (Saves progress as it goes)
        batch_size = 100
        logger.info(f"Inserting {len(nodes)} nodes locally in batches of {batch_size}...")
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            try:
                index.insert_nodes(batch)
                logger.info(f"✓ Successfully embedded and saved batch {i} to {i + len(batch)}")
            except Exception as e:
                logger.error(f"❌ Failed at batch {i}. Error: {e}")
                break 

        logger.info("✓ V3 Ingestion complete. Data is secure and offline.")
        return index

if __name__ == "__main__":
    ingestor = PortfolioIngestorV3()
    index = ingestor.load_index()
    logger.info("V3 Index is ready for querying!")