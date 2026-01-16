import json
import os
import re
import torch  # Necessario per rilevare la GPU
from typing import List, Set, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.embedding_factory import get_shared_embedding_function
# --- Configurazione ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db_data"))

# --- MODIFICA 1: Cambio Modello ---
EMBEDDING_MODEL = "BAAI/bge-m3" 
COLLECTION_NAME = "langchain"

# --- MODIFICA 2: Configurazione GPU per L40 ---
# LangChain usa HuggingFaceEmbeddings che wrappa sentence-transformers.
# Dobbiamo esplicitare il device 'cuda' e la normalizzazione.
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

def _extract_referenced_tables(ddl: str) -> Set[str]:
    """
    Analizza il DDL per trovare i nomi delle tabelle referenziate dalle Foreign Key.
    """
    if not ddl: return set()
    pattern = r'REFERENCES\s+["`]?(\w+)["`]?\s*\('
    matches = re.findall(pattern, ddl, re.IGNORECASE)
    return set(matches)

@tool("search_schema_tool", args_schema=SearchSchemaInput)
def search_schema_tool(query: str, k: int = 15) -> str:
    """
    Retrieval Ibrido: Semantico (BGE-M3) + Relazionale (FK Expansion).
    """
    try:
        vectorstore = get_vectorstore()
        
        # 1. RICERCA SEMANTICA (Anchor Tables)
        # BGE-M3 gestisce query multilingua molto meglio di MiniLM
        anchor_results = vectorstore.similarity_search(query, k=k)
        
        final_schema_map: Dict[str, dict] = {}
        tables_to_fetch_names: Set[str] = set()

        print(f"\n🕸️  [GRAPH RAG] Start: {len(anchor_results)} anchor tables trovate con BGE-M3.")

        # --- FASE A: Processa i risultati semantici ---
        for doc in anchor_results:
            _process_single_doc(doc, final_schema_map, tables_to_fetch_names)

        # --- FASE B: Espansione Relazionale (Fetch Neighbors) ---
        # Questa logica rimane VALIDISSIMA. Anche se l'embedding cambia, 
        # le relazioni SQL (DDL) sono fatti strutturali che il vettore non sostituisce.
        
        
        missing_tables = tables_to_fetch_names - set(final_schema_map.keys())
        
        if missing_tables:
            print(f"🔗 [GRAPH RAG] Espansione: Trovate {len(missing_tables)} tabelle collegate mancanti: {missing_tables}")
            
            try:
                # Accesso diretto alla collezione per ID
                collection = vectorstore._collection
                expansion_results = collection.get(ids=list(missing_tables))
                
                if expansion_results and expansion_results['metadatas']:
                    for i, meta in enumerate(expansion_results['metadatas']):
                        raw_json = meta.get("table_schema")
                        if raw_json:
                            _parse_and_add_to_map(raw_json, final_schema_map)
                            
            except Exception as e:
                print(f"⚠️ Errore nell'espansione FK: {e}")

        # --- FASE C: Output Finale ---
        trimmed_schema = list(final_schema_map.values())
        
        if not trimmed_schema:
            return "NESSUNA TABELLA TROVATA."
            
        json_output = json.dumps(trimmed_schema, indent=2)
        print(f"📊 [GRAPH RAG] Totale tabelle: {len(trimmed_schema)}")
        # Token count approssimativo (utile per debug)
        print(f"📦 Peso Payload: ~{len(json_output)/4:.0f} token")
        
        return json_output

    except Exception as e:
        print(f"❌ CRITICAL ERROR IN TOOL: {e}")
        return f"Error retrieving schema: {str(e)}"

def _process_single_doc(doc, schema_map, tables_to_fetch):
    raw_json = doc.metadata.get("table_schema")
    if raw_json:
        _parse_and_add_to_map(raw_json, schema_map, tables_to_fetch)

def _parse_and_add_to_map(raw_json_str, schema_map, tables_to_fetch=None):
    """Parsing centrale con logica SLIM + Estrazione FK."""
    try:
        full_data = json.loads(raw_json_str)
        tbl_name = full_data.get("table_name")
        
        if tbl_name in schema_map:
            return 

        # 1. Analisi DDL per FK
        ddl = full_data.get("original_ddl", "")
        if tables_to_fetch is not None:
            referenced = _extract_referenced_tables(ddl)
            tables_to_fetch.update(referenced)

        # 2. Creazione Versione SLIM (Ottimizzata per Qwen 32B)
        # BGE-M3 ci permette di trovare la tabella giusta, ma Qwen deve leggerla.
        # Possiamo permetterci descrizioni un po' più lunghe di 300 char.
        
        desc = full_data.get("generated_description", "")
        # Se la descrizione è < 500 char, la teniamo tutta. Qwen 32B ha un'ottima context window.
        if len(desc) > 800: 
            desc = desc[:800] + "... [truncated]"

        # IMPORTANTE: Includiamo i categorical hints estratti nell'ingest
        # Questo è il ponte tra l'analisi euristica e l'LLM finale
        hints = full_data.get("categorical_hints", "") 
        # Solo se i hints sono enormi li tronchiamo, ma sono preziosissimi
        if len(hints) > 1000: 
            hints = hints[:1000] + "... [truncated]"

        # Estraiamo nomi colonne dal DDL per risparmiare token rispetto al DDL intero
        col_names = re.findall(r'(\w+)\s+(?:INTEGER|TEXT|REAL|NUMERIC|BLOB|VARCHAR|CHAR|INT|DATE|TIMESTAMP)', ddl, re.IGNORECASE)

        slim_data = {
            "table": tbl_name,
            "desc": desc,
            "categorical_values": hints, # Campo rinominato per chiarezza all'LLM
            "columns": col_names
        }
        
        schema_map[tbl_name] = slim_data
        
    except Exception as e:
        print(f"Error parsing json: {e}")