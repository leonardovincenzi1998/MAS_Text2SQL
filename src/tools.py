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

# parses raw DDL to find foreign key references missed by metadata
def _extract_referenced_tables(ddl: str) -> Set[str]:
    if not ddl:
        return set()
    
    # robust regex to capture REFERENCES table_name (column)
    pattern = r"""
        REFERENCES\s+
        (
            (?:
                "(?:[^"]+)" |
                `(?:[^`]+)` |
                \[(?:[^\]]+)\] |
                \w+
            )
        )
    """
    matches = re.findall(pattern, ddl, re.IGNORECASE | re.VERBOSE)
    
    normalized = set()
    for m in matches:
        clean_name = m.replace('"', '').replace('`', '').replace('[', '').replace(']', '')
        if '.' in clean_name:
            clean_name = clean_name.split('.')[-1]
        
        canon = get_canonical_name(clean_name)
        if canon:
            normalized.add(canon)
            
    return normalized

# hybrid parser combining JSON and regex implicit FKs
def _parse_and_add_to_map(raw_json_str: str, schema_map: Dict[str, dict], tables_to_fetch: Optional[Set[str]] = None) -> None:
    try:
        full_data = json.loads(raw_json_str)
        
        tbl_name = full_data.get("real_table_name") or full_data.get("table_name")
        if not tbl_name: 
            return

        tbl_canon = get_canonical_name(str(tbl_name))
        if tbl_canon in schema_map: 
            return

        # fk handling combine json and regex
        fk_list = full_data.get("foreign_keys", [])
        existing_targets = {fk.get("to_table_canonical") for fk in fk_list if fk.get("to_table_canonical")}
        
        ddl = full_data.get("original_ddl", "")
        regex_refs = _extract_referenced_tables(ddl)
        
        for ref in regex_refs:
            if ref not in existing_targets:
                fk_list.append({
                    "to_table_canonical": ref,
                    "to_table_real": ref,  
                    "from_column": "inferita_da_ddl",
                    "to_column": "id"
                })
                existing_targets.add(ref)

        if tables_to_fetch is not None:
            tables_to_fetch.update(existing_targets)

        # content handling to reduce LLM token payload
        significant_cols = full_data.get("significant_cols", [])
        if not significant_cols:
             significant_cols = re.findall(r'(\w+)\s+(?:INT|TEXT|REAL|CHAR|DATE)', ddl, re.IGNORECASE)

        # smart description cleanup using regex
        # warning: regex contains italian keywords to safely match LLM output
        desc = full_data.get("generated_description", "")
        
        pattern = (
            r'\n\s*(?:'
            r'\*{0,2}(?:le\s+)?colonn|'
            r'[^\n]*\bcolonn[a-z]*\b[^\n]*:|'
            r'(?:\d+\.|\-)\s*(?:\*\*|`)?\w+(?:\*\*|`)?\s*:|'
            r'\*{0,2}vocabolario|'
            r'\*{0,2}relazioni'
            r')'
        )
        match = re.search(pattern, desc, re.IGNORECASE)
        
        if match:
            desc = desc[:match.start()]
            
        desc = desc.strip()
        if len(desc) > 800: 
            desc = desc[:800] + "..."

        # categorical values cleanup
        raw_profile = full_data.get("data_profile", "")
        categorical_lines = []
        
        for line in raw_profile.split('\n'):
            if line.strip().startswith("- Colonna"):
                if len(line) > 150:
                    line = line[:145] + "...]"
                categorical_lines.append(line.strip())
                
        smart_hints = "\n".join(categorical_lines[:6])
        if len(categorical_lines) > 6: 
            smart_hints += "\n..."
        
        schema_map[tbl_canon] = {
            "table_name": tbl_name,
            "description": desc,
            "columns": significant_cols,   
            "categorical_values": smart_hints,
            "foreign_keys": fk_list,
            "column_samples": full_data.get("column_samples", {})
        }

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
        
        # phase 1: pure similarity search to find anchors
        anchor_results = vectorstore.similarity_search(query, k=k)

        final_schema_map: Dict[str, dict] = {}
        tables_to_fetch_names: Set[str] = set()

        anchor_names = [doc.metadata.get("canonical_name", "unknown") for doc in anchor_results]
        print(f"\n🕸️  [GRAPH RAG] Start: {len(anchor_results)} anchor tables found: {anchor_names}")

        # phase a: process anchor nodes
        for doc in anchor_results:
            _process_single_doc(doc, final_schema_map, tables_to_fetch_names)

        # phase b: expansion fetching neighbors
        missing_tables = tables_to_fetch_names - set(final_schema_map.keys())

        if missing_tables:
            print(f"🔗 [GRAPH RAG] Expansion: Fetching {len(missing_tables)} linked tables: {list(missing_tables)[:5]}...")
            try:
                collection = vectorstore._collection
                expansion_results = collection.get(ids=list(missing_tables))
                
                if expansion_results and expansion_results.get("metadatas"):
                    for meta in expansion_results["metadatas"]:
                        raw_json = meta.get("table_schema")
                        if raw_json:
                            _parse_and_add_to_map(raw_json, final_schema_map, tables_to_fetch_names)
            except Exception as e:
                print(f"⚠️ Expansion error: {e}")

        # phase c: output formatting
        trimmed_schema = list(final_schema_map.values())
        if not trimmed_schema:
            return empty_json

        json_output = json.dumps(trimmed_schema, indent=2)
        
        # debugging payload size
        print(f"📊 [GRAPH RAG] Total tables: {len(trimmed_schema)} | Estimated payload tokens: {len(json_output)//4}")
        
        with open("debug_payload_v2.json", "w", encoding="utf-8") as f:
            f.write(json_output)
        print("💾 V2 Payload saved to 'debug_payload_v2.json'")

        return json_output

    except Exception as e:
        print(f"❌ CRITICAL ERROR IN TOOL: {e}")
        return empty_json