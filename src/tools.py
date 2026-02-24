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
        
        # Torniamo alla similarity search pura
        anchor_results = vectorstore.similarity_search(query, k=k)

        final_schema_map: Dict[str, dict] = {}
        tables_to_fetch_names: Set[str] = set()

        # 🔥 Estraiamo i nomi canonici dai metadati per il log
        anchor_names = [doc.metadata.get("canonical_name", "unknown") for doc in anchor_results]
        print(f"\n🕸️  [GRAPH RAG] Start: {len(anchor_results)} anchor tables trovate: {anchor_names}")

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
    HYBRID PARSER DEFINITIVO:
    Mantiene le FK ricche (con colonne) e integra le FK implicite trovate via Regex.
    """
    try:
        full_data = json.loads(raw_json_str)
        
        tbl_name = full_data.get("real_table_name") or full_data.get("table_name")
        if not tbl_name: return

        tbl_canon = get_canonical_name(str(tbl_name))
        if tbl_canon in schema_map: return

        # --- 1. GESTIONE FK: UNIONE JSON + REGEX ---
        fk_list = full_data.get("foreign_keys", [])
        
        # Raccogliamo i target già noti dai metadati
        existing_targets = {fk.get("to_table_canonical") for fk in fk_list if fk.get("to_table_canonical")}
        
        # Usiamo il fallback Regex sul DDL per trovare relazioni sfuggite
        ddl = full_data.get("original_ddl", "")
        regex_refs = _extract_referenced_tables(ddl)
        
        # Aggiungiamo eventuali FK trovate dalla regex che non erano nei metadati
        for ref in regex_refs:
            if ref not in existing_targets:
                fk_list.append({
                    "to_table_canonical": ref,
                    "to_table_real": ref,  
                    "from_column": "inferita_da_ddl",
                    "to_column": "id"
                })
                existing_targets.add(ref)

        # Aggiorniamo la lista globale di espansione per far scaricare a Chroma i vicini
        if tables_to_fetch is not None:
            tables_to_fetch.update(existing_targets)


        # --- 2. GESTIONE CONTENUTO (DIETA PER LLM) ---
        significant_cols = full_data.get("significant_cols", [])
        if not significant_cols:
             significant_cols = re.findall(r'(\w+)\s+(?:INT|TEXT|REAL|CHAR|DATE)', ddl, re.IGNORECASE)

        raw_profile = full_data.get("data_profile", "")
        categorical_lines = [line.strip() for line in raw_profile.split('\n') if line.strip().startswith("- Colonna")]
        smart_hints = "\n".join(categorical_lines[:6])
        if len(categorical_lines) > 6: smart_hints += "\n..."

      # --- PULIZIA INTELLIGENTE DELLA DESCRIZIONE (SUPER REGEX) ---
        desc = full_data.get("generated_description", "")
        
        # Intercetta in modo case-insensitive:
        # 1. Frasi che iniziano con "Colonne" o "Le colonne" (con 0, 1 o 2 asterischi)
        # 2. Qualsiasi riga che contiene la parola "colonn" e finisce con i due punti ":"
        # 3. Elenchi numerati (es. "1. **IdMobile**:") o puntati (es. "- **IdLocale**:")
        # 4. Sezioni "Vocabolario" o "Relazioni"
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

        # --- PULIZIA VALORI CATEGORICI (TAGLIA I MURI DI TESTO) ---
        raw_profile = full_data.get("data_profile", "")

        categorical_lines = []
        for line in raw_profile.split('\n'):
            if line.strip().startswith("- Colonna"):
                # Se la stringa dei valori è chilometrica (es. Annotazioni), la tronchiamo
                if len(line) > 150:
                    line = line[:145] + "...]"
                categorical_lines.append(line.strip())
                
        smart_hints = "\n".join(categorical_lines[:6])
        if len(categorical_lines) > 6: 
            smart_hints += "\n..."
        
        # Costruzione Oggetto Finale con FK complete!
        schema_map[tbl_canon] = {
            "table_name": tbl_name,
            "description": desc,
            "columns": significant_cols,   
            "categorical_values": smart_hints,
            "foreign_keys": fk_list,
            "column_samples": full_data.get("column_samples", {})
        }

    except Exception as e:
        print(f"Error parsing json: {e}")