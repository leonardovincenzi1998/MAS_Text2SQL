import json
import os
import re
import torch
from typing import List, Set, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_chroma import Chroma
from src.models import SearchSchemaInput
from src.embedding_factory import get_shared_embedding_function
from src.utils import get_canonical_name
from src.config import CHROMA_PATH, COLLECTION_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔌 Loading Embeddings on: {DEVICE}")

# cache the embedding function globally for the module
EMBEDDING_FUNCTION = get_shared_embedding_function()

# initializes and returns the Chroma vector store connection
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDING_FUNCTION,
        collection_name=COLLECTION_NAME
    )

# hybrid parser combining JSON and regex implicit FKs
def _parse_and_add_to_map(raw_json_str: str, schema_map: Dict[str, dict], tables_to_fetch: Optional[Set[str]] = None) -> None:
    """
    Parser istantaneo: i dati sono già stati puliti, formattati e "snelliti" 
    in fase di ingestion. Dobbiamo solo fare il load e mappare le FK.
    """
    try:
        full_data = json.loads(raw_json_str)
        
        tbl_name = full_data.get("table_name")
        if not tbl_name: 
            return

        tbl_canon = get_canonical_name(str(tbl_name))
        
        # Evita duplicati
        if tbl_canon in schema_map: 
            return

        # Popoliamo il set delle tabelle da espandere tramite le Foreign Keys
        if tables_to_fetch is not None:
            fk_list = full_data.get("foreign_keys", [])
            for fk in fk_list:
                target_canon = fk.get("to_table_canonical")
                if target_canon:
                    tables_to_fetch.add(target_canon)

        # Salviamo lo "Slim Schema" direttamente (non contiene più il DDL pesante)
        schema_map[tbl_canon] = full_data

    except Exception as e:
        print(f"Error parsing json in _parse_and_add_to_map: {e}")

# wrapper to process a single Chroma document and its metadata
def _process_single_doc(doc, schema_map: Dict[str, dict], tables_to_fetch: Set[str]) -> None:
    raw_json = doc.metadata.get("table_schema")
    if raw_json:
        _parse_and_add_to_map(raw_json, schema_map, tables_to_fetch)

# hybrid retrieval tool combining semantic search and regex fk expansion
@tool("search_schema_tool", args_schema=SearchSchemaInput)
def search_schema_tool(query: str, k: int = 10) -> str:
    """Ricerca lo schema del database utilizzando ricerca semantica ibrida e foreign keys."""
    empty_json = json.dumps([], indent=2)

    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        
        # --- FIX PUNTO 3: CLAMP DEL PARAMETRO k ---
        # Evita di chiedere a ChromaDB più tabelle di quante ne esistano realmente
        total_docs = collection.count()
        actual_k = min(k, total_docs) if total_docs > 0 else k

        # Fase 1: Ricerca puramente semantica (trova le "Anchor Tables")
        anchor_results = vectorstore.similarity_search(query, k=actual_k)

        final_schema_map: Dict[str, dict] = {}
        tables_to_fetch_names: Set[str] = set()

        anchor_names = [doc.metadata.get("canonical_name", "unknown") for doc in anchor_results]
        print(f"\n🕸️  [GRAPH RAG] Start: {len(anchor_results)} anchor tables found: {anchor_names}")

        # Fase A: Processiamo i nodi àncora
        for doc in anchor_results:
            _process_single_doc(doc, final_schema_map, tables_to_fetch_names)

        # Fase B: Espansione (Fetch dei vicini tramite FK)
        missing_tables = tables_to_fetch_names - set(final_schema_map.keys())

        if missing_tables:
            print(f"🔗 [GRAPH RAG] Expansion: Fetching {len(missing_tables)} linked tables: {list(missing_tables)[:5]}...")
            try:
                expansion_results = collection.get(ids=list(missing_tables))
                
                if expansion_results and expansion_results.get("metadatas"):
                    for meta in expansion_results["metadatas"]:
                        raw_json = meta.get("table_schema")
                        if raw_json:
                            _parse_and_add_to_map(raw_json, final_schema_map, tables_to_fetch_names)
            except Exception as e:
                print(f"⚠️ Expansion error: {e}")

        # Fase C: Output Formatting
        trimmed_schema = list(final_schema_map.values())
        if not trimmed_schema:
            return empty_json

        json_output = json.dumps(trimmed_schema, indent=2)
        
        # Debugging payload size
        print(f"📊 [GRAPH RAG] Total tables: {len(trimmed_schema)} | Estimated payload tokens: {len(json_output)//4}")
        
        with open("debug_payload_v2.json", "w", encoding="utf-8") as f:
            f.write(json_output)
        print("💾 V2 Payload saved to 'debug_payload_v2.json'")

        return json_output

    except Exception as e:
        print(f"❌ CRITICAL ERROR IN TOOL: {e}")
        return empty_json