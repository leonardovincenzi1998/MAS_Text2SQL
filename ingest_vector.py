import sqlite3
import json
import argparse
import asyncio
import os    
import chromadb
from chromadb.utils import embedding_functions
from src.database import DatabaseManager
from src.config import get_model
from pydantic_ai import Agent

# --- CONFIGURAZIONE DEFAULT (Sovrascritta dagli argomenti) ---
# Usiamo percorsi relativi sicuri come fallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "cloneDefinitivoDB.db")
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_data")

# IMPORTANTE: LangChain di default usa la collection "langchain". 
# Usiamo questo nome per garantire compatibilità immediata con tools.py
COLLECTION_NAME = "langchain"

# --- PROMPT AGGIORNATO (Context-Aware) ---
DESCRIPTION_AGENT_PROMPT = """
You are an expert Data Steward. Your task is to generate semantic documentation for an SQL table.

You will receive:
1. The DDL schema (Create Table).
2. A sample of 3 rows of real data from the table.

You must produce a discursive description (in Italian) explaining:
1. What the main entity represents (e.g. “Customers”).
2. The meaning of the columns, BASED ON THE SAMPLE DATA. If a column has a cryptic name but the data is clearly dates or amounts, describe it based on the data.
3. If the data are incomprehensible codes and you cannot deduce their meaning, write ‘Technical column of unknown meaning’ instead of inventing something.

WARNING: Do not hallucinate meanings that are not supported by the data or the column name.

Required output: ONLY the text of the description.
"""

def get_table_samples(db_path: str, table_name: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not rows:
            return "Nessun dato presente."
            
        sample_text = f"Colonne: {', '.join(col_names)}\n"
        for i, row in enumerate(rows):
            sample_text += f"Riga {i+1}: {str(row)}\n"
        return sample_text
    except Exception as e:
        return f"Errore campioni: {e}"

async def generate_table_description(ddl_text: str, samples_text: str) -> str:
    agent = Agent(
        model=get_model(),
        system_prompt=DESCRIPTION_AGENT_PROMPT
    )
    user_content = f"--- DDL ---\n{ddl_text}\n--- SAMPLES ---\n{samples_text}"
    result = await agent.run(user_content)
    return result.output

async def main():
    # --- 1. Parsing Argomenti (Per funzionare bene con SLURM) ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=os.getenv("DB_PATH", DEFAULT_DB_PATH))
    parser.add_argument("--chroma_path", default=os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH))
    args = parser.parse_args()

    db_path = args.db_path
    chroma_path = args.chroma_path

    if not os.path.exists(db_path):
        print(f"❌ Errore: Database non trovato in: {db_path}")
        return

    print(f"🔌 DB SQL: {db_path}")
    print(f"💾 Vector DB: {chroma_path}")

    db_manager = DatabaseManager(db_path)
    tables = db_manager.search_tables(None)
    
    # --- 2. Setup Chroma con Embeddings COMPATIBILI ---
    # Usiamo SentenceTransformer esplicitamente per matchare 'all-MiniLM-L6-v2' di tools.py
    print("🧠 Caricamento funzione di embedding...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    # Reset della collezione per evitare duplicati
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("🧹 Collezione esistente rimossa.")
    except:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    print("\n🚀 Inizio Arricchimento Semantico...")

    for table in tables:
        print(f"   👉 Processando: {table}...")
        
        ddl = db_manager.get_table_ddl(table)
        samples = get_table_samples(db_path, table)
        
        # Generazione descrizione con LLM (Qwen)
        description = await generate_table_description(ddl, samples)
        
        # --- 3. SALVATAGGIO FORMATO COMPATIBILE CON TOOLS.PY ---
        # tools.py si aspetta: json.loads(doc.metadata.get("table_schema"))
        # Quindi dobbiamo salvare un JSON dentro il campo 'table_schema'.
        
        metadata_payload = {
            "table_name": table,
            "original_ddl": ddl,
            "generated_description": description
        }

        collection.add(
            documents=[description],
            metadatas=[{
                "table_name": table,
                # IL TRUCCO: Salviamo il json come stringa dentro un campo specifico
                "table_schema": json.dumps(metadata_payload) 
            }],
            ids=[table]
        )

    print(f"\n✅ Ingestion Completata! {collection.count()} tabelle indicizzate.")

if __name__ == "__main__":
    asyncio.run(main())