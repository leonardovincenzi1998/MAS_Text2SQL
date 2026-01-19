import json
import os
import re
import torch  # Necessario per rilevare la GPU
from typing import List, Set, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.embedding_factory import get_shared_embedding_function
from src.naming import get_canonical_name

# --- Configurazione ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db_data"))

EMBEDDING_MODEL = "BAAI/bge-m3"
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
# NORMALIZZAZIONE + MAPPA canonical_name -> real_id (CHROMA)
# =============================================================================


def _extract_referenced_tables(ddl: str) -> Set[str]:
    """
    Analizza il DDL per trovare i nomi delle tabelle referenziate dalle Foreign Key.
    Gestisce:
    - quoting: "TAB", `TAB`, [TAB]
    - schema-qualified: schema.TAB
    Restituisce SEMPRE nomi normalizzati (lower, no schema).
    """
    if not ddl:
        return set()

    # Cattura un identificatore singolo o schema.qualificato, con quote/brackets opzionali
    # Esempi catturati:
    #   REFERENCES "SCHEMA"."TBL"(
    #   REFERENCES schema.tbl(
    #   REFERENCES `tbl`(
    #   REFERENCES [tbl](
    pattern = r"""
        REFERENCES\s+
        (
            (?:
                "(?:[^"]+)" |        # "name"
                `(?:[^`]+)` |        # `name`
                \[(?:[^\]]+)\] |     # [name]
                \w+                  # bare
            )
            (?:
                \s*\.\s*
                (?:
                    "(?:[^"]+)" |
                    `(?:[^`]+)` |
                    \[(?:[^\]]+)\] |
                    \w+
                )
            )?
        )
        \s*\(
    """
    matches = re.findall(pattern, ddl, re.IGNORECASE | re.VERBOSE)

    normalized = set()
    for m in matches:
        normalized.add(get_canonical_name(m))
    return {x for x in normalized if x}


@tool("search_schema_tool", args_schema=SearchSchemaInput)
def search_schema_tool(query: str, k: int = 15) -> str:
    """
    Retrieval Ibrido: Semantico (BGE-M3) + Relazionale (FK Expansion).
    """
    empty_json = json.dumps([], indent=2)

    try:
        vectorstore = get_vectorstore()

        # 1. RICERCA SEMANTICA (Anchor Tables)
        anchor_results = vectorstore.similarity_search(query, k=k)

        # ATTENZIONE: ora indicizziamo schema_map per NOME CANONICO
        final_schema_map: Dict[str, dict] = {}
        tables_to_fetch_names: Set[str] = set()

        print(f"\n🕸️  [GRAPH RAG] Start: {len(anchor_results)} anchor tables trovate con BGE-M3.")

        # --- FASE A: Processa i risultati semantici ---
        for doc in anchor_results:
            _process_single_doc(doc, final_schema_map, tables_to_fetch_names)

        # --- FASE B: Espansione Relazionale (Fetch Neighbors) ---
        missing_tables = tables_to_fetch_names - set(final_schema_map.keys())

        if missing_tables:
            print(f"🔗 [GRAPH RAG] Espansione: Trovate {len(missing_tables)} tabelle collegate mancanti: {missing_tables}")

            # Traduci canonical -> real_id per Chroma.get(ids=...)
            try:
                collection = vectorstore._collection
                expansion_results = collection.get(ids=list(missing_tables))

                if expansion_results and expansion_results.get("metadatas"):
                    for meta in expansion_results["metadatas"]:
                        raw_json = meta.get("table_schema")
                        if raw_json:
                            _parse_and_add_to_map(raw_json, final_schema_map, tables_to_fetch_names)

            except Exception as e:
                print(f"⚠️ Errore nell'espansione FK: {e}")


        # --- FASE C: Output Finale ---
        trimmed_schema = list(final_schema_map.values())

        if not trimmed_schema:
            print("ℹ️ [GRAPH RAG] Nessuna tabella trovata.")
            return empty_json

        json_output = json.dumps(trimmed_schema, indent=2)
        print(f"📊 [GRAPH RAG] Totale tabelle: {len(trimmed_schema)}")
        print(f"📦 Peso Payload: ~{len(json_output)/4:.0f} token")

        return json_output

    except Exception as e:
        print(f"❌ CRITICAL ERROR IN TOOL: {e}")
        return empty_json


def _process_single_doc(doc, schema_map, tables_to_fetch):
    raw_json = doc.metadata.get("table_schema")
    if raw_json:
        _parse_and_add_to_map(raw_json, schema_map, tables_to_fetch)


def _parse_and_add_to_map(raw_json_str, schema_map, tables_to_fetch=None):
    """Parsing centrale con logica SLIM + Estrazione FK. (Ora con nomi normalizzati)"""
    try:
        full_data = json.loads(raw_json_str)
        tbl_name = full_data.get("real_table_name") or full_data.get("table_name")

        fk_list = full_data.get("foreign_keys", [])

        if not tbl_name:
            return

        tbl_canon = get_canonical_name(str(tbl_name))
        if not tbl_canon:
            return

        # schema_map ora è indicizzata per nome CANONICO
        if tbl_canon in schema_map:
            return

        # 1. Analisi DDL per FK (normalizzate)
        ddl = full_data.get("original_ddl", "")
        if tables_to_fetch is not None:
            fk_list = full_data.get("foreign_keys") or []
            if fk_list:
                referenced = set()
                for fk in fk_list:
                    to_tbl = fk.get("to_table_canonical") or fk.get("to_table_real") or ""
                    canon = get_canonical_name(str(to_tbl))
                    if canon:
                        referenced.add(canon)
                tables_to_fetch.update(referenced)
            else:
                referenced = _extract_referenced_tables(ddl)
                tables_to_fetch.update(referenced)

        # 2. Creazione Versione SLIM
        desc = full_data.get("generated_description", "")
        if len(desc) > 800:
            desc = desc[:800] + "... [truncated]"

        hints = full_data.get("categorical_hints", "")
        if len(hints) > 1000:
            hints = hints[:1000] + "... [truncated]"

        col_names = re.findall(
            r'(\w+)\s+(?:INTEGER|TEXT|REAL|NUMERIC|BLOB|VARCHAR|CHAR|INT|DATE|TIMESTAMP)',
            ddl,
            re.IGNORECASE
        )

        slim_data = {
            "table": tbl_name,              # nome originale (utile all'agente SQL)
            "table_canonical": tbl_canon,   # nome canonico (debug/consistenza)
            "desc": desc,
            "categorical_values": hints,
            "columns": col_names,
            "foreign_keys": fk_list
        }

        schema_map[tbl_canon] = slim_data

    except Exception as e:
        print(f"Error parsing json: {e}")
