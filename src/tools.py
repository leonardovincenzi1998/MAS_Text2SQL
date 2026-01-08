import json
import os
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configurazione ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
#CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_data")   locale

# Cerca prima la variabile d'ambiente (impostata da SLURM), 
# altrimenti usa il percorso locale di default.
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db_data"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # DEVE essere lo stesso modello

# --- 1. Definizione dell'Input con Pydantic ---
class SearchSchemaInput(BaseModel):
    """Input schema per il tool di ricerca nel database."""
    query: str = Field(
        description="Le parole chiave o la descrizione in linguaggio naturale dei dati che stai cercando (es. 'vincoli paesaggistici', 'anagrafica clienti')."
    )
    k: int = Field(
        default=15,
        description="Il numero di tabelle simili da recuperare. Default è 10."
    )

# --- 2. Inizializzazione Risorse (Lazy Loading) ---
# È meglio inizializzare l'embedding function fuori dalla funzione per non ricaricarla ogni volta
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore():
    """Connette al DB Chroma esistente."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

# --- 3. Il Tool vero e proprio ---
@tool("search_schema_tool", args_schema=SearchSchemaInput)
def search_schema_tool(query: str, k: int = 15) -> str:
    """
    Cerca nel database vettoriale le tabelle più rilevanti basandosi sulla query.
    Restituisce uno schema JSON ridotto ('Trimmed Schema') contenente solo le tabelle utili.
    """
    try:
        vectorstore = get_vectorstore()
        
        # Esegue la ricerca semantica
        results = vectorstore.similarity_search(query, k=k)
        
        # --- 🛠️ BLOCCO DEBUG START 🛠️ ---
        print(f"\n🔎 [DEBUG VECTOR DB] Ho trovato {len(results)} tabelle candidate per la query: '{query}'")
        found_tables = []
        for i, doc in enumerate(results):
        # Recupera il nome tabella dai metadati (adatta la chiave se diversa, es. 'table_name')
            tbl_name = doc.metadata.get("table_name", "N/A")
            found_tables.append(tbl_name)
            print(f"   {i+1}. {tbl_name} (Score/Metadata: {doc.metadata})")
    
        print(f"👀 Lista completa inviata all'LLM: {found_tables}")
        print("-" * 50)
        # --- 🛠️ BLOCCO DEBUG END 🛠️ ---

        if not results:
            return "Nessuna tabella trovata pertinente alla tua ricerca."

        trimmed_schema = []
        
        # Costruiamo il JSON di risposta estraendo i metadati
        for doc in results:
            # I metadati contengono il JSON della tabella che abbiamo salvato prima
            table_info = json.loads(doc.metadata.get("table_schema", "{}"))
            trimmed_schema.append(table_info)
            
        # Formattiamo come stringa JSON per l'LLM
        return json.dumps(trimmed_schema, indent=2)

    except Exception as e:
        return f"Errore durante la ricerca nello schema: {str(e)}"