"""
ingest_v2.py — Step 2: Table-Aware Chunking & Embedding
=========================================================

Uses MarkdownElementNodeParser to separate tables from text,
ensuring tables remain fully intact in ChromaDB.
"""

import os
import json
import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("PortfolioIngestor_V2")

load_dotenv()

# NEW PATHS
CACHE_FILE = Path("./data/parsed_docs_v2.json")
CHROMA_DIR = Path("./chroma_db_v2")
CHROMA_COLLECTION = "apple_10k_v2"


class PortfolioIngestorV2:
    def __init__(self):
        self._validate_env()
        self._configure_settings()
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        logger.info("ChromaDB V2 client initialised at '%s'", CHROMA_DIR)

    def _validate_env(self):
        required = ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise EnvironmentError(
                f"Missing required env variables: {', '.join(missing)}."
            )

    def _configure_settings(self):
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=os.environ["OPENROUTER_API_KEY"],
            api_base=os.environ["OPENROUTER_BASE_URL"],
        )

        Settings.llm = OpenRouter(
            model="meta-llama/llama-3.3-70b-instruct:free",
            api_key=os.environ["OPENROUTER_API_KEY"],
            api_base=os.environ["OPENROUTER_BASE_URL"],
            is_function_calling_model=True,
        )

        logger.info(
            "LLM configured for strict JSON function calling via OpenRouter."
        )

    def _load_cached_documents(self) -> list:
        if not CACHE_FILE.exists():
            raise FileNotFoundError(
                f"Cache file {CACHE_FILE} not found. Run parse_v2.py first."
            )

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            docs_dicts = json.load(f)

        documents = [Document.from_dict(d) for d in docs_dicts]
        logger.info(
            "Loaded %d document sections from V2 cache.", len(documents)
        )
        return documents

    # def _chunk_documents(self, documents: list) -> list:
    #     logger.info("Stage 1: Table-Aware Markdown Parsing…")
    #
    #     element_parser = MarkdownElementNodeParser(
    #         llm=Settings.llm,
    #         num_workers=1,
    #         skip_table_summaries=True  # <--- THIS FIXES THE CRASH
    #     )
    #
    #     # We extract base nodes (text) and object nodes (tables)
    #     nodes = element_parser.get_nodes_from_documents(documents)
    #
    #     base_nodes, objects = element_parser.get_nodes_and_objects(nodes)
    #     logger.info(
    #         " → Extracted %d base text nodes and %d tabular objects.",
    #         len(base_nodes), len(objects)
    #     )
    #
    #     # Sentence-split text nodes, leave tables intact
    #     logger.info("Stage 2: Sentence splitting the text nodes…")
    #     sentence_splitter = SentenceSplitter(
    #         chunk_size=1024,
    #         chunk_overlap=128
    #     )
    #
    #     final_text_nodes = sentence_splitter.get_nodes_from_documents(
    #         base_nodes
    #     )
    #
    #     final_nodes = final_text_nodes + objects
    #     logger.info(
    #         " → %d final nodes ready for embedding.",
    #         len(final_nodes)
    #     )
    #     return final_nodes

    def _chunk_documents(self, documents: list) -> list:
        logger.info(
            "Stage 1: Parsing Markdown structure locally (No API calls)…"
        )

        # UPGRADE: Switched to the local Markdown parser.
        # Groups tables and text by Markdown headers natively
        from llama_index.core.node_parser import MarkdownNodeParser

        md_parser = MarkdownNodeParser()

        md_nodes = md_parser.get_nodes_from_documents(
            documents,
            show_progress=True
        )
        logger.info(" → %d Markdown nodes extracted.", len(md_nodes))

        logger.info("Stage 2: Sentence splitting…")

        sentence_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=128
        )

        final_nodes = sentence_splitter.get_nodes_from_documents(
            md_nodes,
            show_progress=True
        )
        logger.info(
            " → %d final nodes ready for embedding.",
            len(final_nodes)
        )

        return final_nodes

    def load_index(self) -> VectorStoreIndex:
        collection = self.chroma_client.get_or_create_collection(
            CHROMA_COLLECTION
        )
        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        if collection.count() > 0:
            logger.info("Existing V2 collection found. Loading index…")
            return VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )

        logger.info(
            "No existing collection found. Starting V2 ingestion pipeline…"
        )

        documents = self._load_cached_documents()
        nodes = self._chunk_documents(documents)

        logger.info(
            "Embedding and persisting %d nodes to ChromaDB V2…",
            len(nodes)
        )

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )

        logger.info("✓ V2 Ingestion complete. Index persisted.")
        return index


if __name__ == "__main__":
    ingestor = PortfolioIngestorV2()
    index = ingestor.load_index()
    logger.info("V2 Index is ready for querying!")