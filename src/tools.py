import json
import os
import re
import torch
from typing import List, Set, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_chroma import Chroma
from src.embedding_factory import get_shared_embedding_function
from src.naming import get_canonical_name

# --- Configurazione ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db_data"))
COLLECTION_NAME = "langchain"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔌 Loading Embeddings on: {device}")

embedding_function = get_shared_embedding_function()

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

class SearchSchemaInput(BaseModel):
    query: str = Field(description="Entità o keywords da cercare.")
    k: int = Field(default=10, description="Numero tabelle 'anchor' semantiche.")

# =============================================================================
# HELPERS (IL SEGRETO DEL SUCCESSO DELLA VECCHIA VERSIONE)
# =============================================================================

def _extract_referenced_tables(ddl: str) -> Set[str]:
    """
    Analizza il DDL grezzo per trovare riferimenti FK che i metadati potrebbero aver perso.
    """
    if not ddl:
        return set()
    
    # Regex robusta per catturare 'REFERENCES tabella (colonna)'
    pattern = r"""
        REFERENCES\s+
        (
            (?:
                "(?:[^"]+)" |        # "name"
                `(?:[^`]+)` |        # `name`
                \[(?:[^\]]+)\] |     # [name]
                \w+                  # bare name
            )
        )
    """
    matches = re.findall(pattern, ddl, re.IGNORECASE | re.VERBOSE)
    
    normalized = set()
    for m in matches:
        # Pulisce eventuali quote o schema prefix rimasti
        clean_name = m.replace('"', '').replace('`', '').replace('[', '').replace(']', '')
        if '.' in clean_name:
            clean_name = clean_name.split('.')[-1]
        
        canon = get_canonical_name(clean_name)
        if canon:
            normalized.add(canon)
            
    return normalized

# =============================================================================
# TOOL
# =============================================================================

@tool("search_schema_tool", args_schema=SearchSchemaInput)
def search_schema_tool(query: str, k: int = 10) -> str:
    """Retrieval Ibrido: Semantico + Regex FK Expansion."""
    empty_json = json.dumps([], indent=2)

    try:
        vectorstore = get_vectorstore()
        
        # Usiamo MMR per diversificare, ma con fetch_k alto per scansionare bene
        anchor_results = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=30, lambda_mult=0.6
        )

        final_schema_map: Dict[str, dict] = {}
        tables_to_fetch_names: Set[str] = set()

        print(f"\n🕸️  [GRAPH RAG] Start: {len(anchor_results)} anchor tables trovate.")

        # --- FASE A: Anchor Nodes ---
        for doc in anchor_results:
            _process_single_doc(doc, final_schema_map, tables_to_fetch_names)

        # --- FASE B: Expansion (Neighbors) ---
        missing_tables = tables_to_fetch_names - set(final_schema_map.keys())

        if missing_tables:
            print(f"🔗 [GRAPH RAG] Espansione: Recupero {len(missing_tables)} tabelle collegate: {list(missing_tables)[:5]}...")
            try:
                collection = vectorstore._collection
                expansion_results = collection.get(ids=list(missing_tables))
                if expansion_results and expansion_results.get("metadatas"):
                    for meta in expansion_results["metadatas"]:
                        raw_json = meta.get("table_schema")
                        if raw_json:
                            _parse_and_add_to_map(raw_json, final_schema_map, tables_to_fetch_names)
            except Exception as e:
                print(f"⚠️ Errore expansion: {e}")

        # --- FASE C: Output ---
        trimmed_schema = list(final_schema_map.values())
        if not trimmed_schema:
            return empty_json

        # Debug peso
        json_output = json.dumps(trimmed_schema, indent=2)
        print(f"📊 [GRAPH RAG] Totale tabelle: {len(trimmed_schema)} | Payload Token stimati: {len(json_output)//4}")
        
        with open("debug_payload_v2.json", "w", encoding="utf-8") as f:
            f.write(json_output)
        print("💾 Payload V2 salvato in 'debug_payload_v2.json'")
        
        return json_output

    except Exception as e:
        print(f"❌ CRITICAL ERROR IN TOOL: {e}")
        return empty_json


def _process_single_doc(doc, schema_map, tables_to_fetch):
    raw_json = doc.metadata.get("table_schema")
    if raw_json:
        _parse_and_add_to_map(raw_json, schema_map, tables_to_fetch)


def _parse_and_add_to_map(raw_json_str, schema_map, tables_to_fetch=None):
    """
    HYBRID PARSER:
    1. Usa i Smart Hints (Valori Categorici).
    2. Usa il Regex Fallback per le FK (per trovare le tabelle nascoste).
    3. Tronca le descrizioni (per evitare il crash LLM).
    """
    try:
        full_data = json.loads(raw_json_str)
        
        tbl_name = full_data.get("real_table_name") or full_data.get("table_name")
        if not tbl_name: return

        tbl_canon = get_canonical_name(str(tbl_name))
        if tbl_canon in schema_map: return

        # --- 1. GESTIONE FK (LA PARTE CHE MANCAVA) ---
        # Prima proviamo i metadati puliti
        fk_list = full_data.get("foreign_keys", [])
        referenced = set()
        
        # Aggiungiamo da JSON
        for fk in fk_list:
            target = fk.get("to_table_canonical")
            if target: referenced.add(target)

        # POI IL FALLBACK REGEX (Cruciale!)
        ddl = full_data.get("original_ddl", "")
        regex_refs = _extract_referenced_tables(ddl)
        referenced.update(regex_refs)

        # Aggiorniamo la lista globale di espansione
        if tables_to_fetch is not None:
            tables_to_fetch.update(referenced)


        # --- 2. GESTIONE CONTENUTO (DIETA PER LLM) ---
        
        # Colonne: Preferiamo quelle filtrate, ma fallback su regex se mancano
        significant_cols = full_data.get("significant_cols", [])
        if not significant_cols:
             # Fallback vecchia scuola se l'ingestion non ha funzionato
             significant_cols = re.findall(r'(\w+)\s+(?:INT|TEXT|REAL|CHAR|DATE)', ddl, re.IGNORECASE)

        # Smart Hints (Valori Categorici) - Estraiamo solo le righe utili
        raw_profile = full_data.get("data_profile", "")
        categorical_lines = [line.strip() for line in raw_profile.split('\n') if line.strip().startswith("- Colonna")]
        smart_hints = "\n".join(categorical_lines[:6]) # Max 6 righe
        if len(categorical_lines) > 6: smart_hints += "\n..."

        # Descrizione: TRONCAMENTO AGGRESSIVO (Per evitare il Loop)
        desc = full_data.get("generated_description", "")
        if len(desc) > 350: 
            desc = desc[:350] + "..."

        # Costruzione Oggetto Finale
        schema_map[tbl_canon] = {
            "table_name": tbl_name,
            "description": desc,
            "columns": significant_cols,   
            "categorical_values": smart_hints,
            # Passiamo le FK trovate (miste json/regex) all'LLM per aiutarlo a capire i link
            "foreign_keys": list(referenced) 
        }

    except Exception as e:
        print(f"Error parsing json: {e}")