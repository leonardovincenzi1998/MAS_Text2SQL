import asyncio
import os
import sqlite3 # Usiamo sqlite3 diretto per semplicità nello script di ingest
import chromadb
from src.database import DatabaseManager
from src.config import get_model
from pydantic_ai import Agent

# --- CONFIGURAZIONE ---
DB_PATH = "C:\\Users\\lvincenzi\\Tesi\\Terreni_Fabbricati.db" 
CHROMA_PATH = "./chroma_db_data"
COLLECTION_NAME = "sql_schema_metadata"

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
    """
    Extracts 3 sample rows from the table to provide context to the LLM.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Prende 3 righe a caso o le prime 3
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        rows = cursor.fetchall()
        
        # Recupera anche i nomi delle colonne per formattare bene
        col_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not rows:
            return "Nessun dato presente nella tabella."
            
        # Formatta come testo leggibile
        sample_text = f"Colonne: {', '.join(col_names)}\n"
        for i, row in enumerate(rows):
            sample_text += f"Riga {i+1}: {str(row)}\n"
            
        return sample_text
    except Exception as e:
        return f"Impossibile recuperare campioni: {e}"

async def generate_table_description(ddl_text: str, samples_text: str) -> str:
    """
    Uses the LLM to generate a semantic description using DDL + Data.
    """
    agent = Agent(
        model=get_model(),
        system_prompt=DESCRIPTION_AGENT_PROMPT
    )
    
    user_content = f"""
    --- DDL TABELLA ---
    {ddl_text}
    
    --- DATI CAMPIONE (Context) ---
    {samples_text}
    """
    
    result = await agent.run(user_content)
    return result.output

async def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Errore: Database '{DB_PATH}' non trovato.")
        return

    print(f"🔌 Connessione al DB SQLite: {DB_PATH}...")
    db_manager = DatabaseManager(DB_PATH)
    tables = db_manager.search_tables(None)
    
    # Inizializzazione Chroma
    print(f"💾 Inizializzazione ChromaDB in: {CHROMA_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Pulizia (Opzionale)
    if collection.count() > 0:
        print("🧹 Pulizia vecchia collezione...")
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    print("\n🚀 Inizio Arricchimento Semantico (DDL + Dati)...")

    for table in tables:
        print(f"   👉 Processando tabella: {table}...")
        
        # 1. DDL
        ddl = db_manager.get_table_ddl(table)
        
        # 2. CAMPIONI (La protezione contro le allucinazioni)
        samples = get_table_samples(DB_PATH, table)
        
        # 3. GENERAZIONE DESCRIZIONE
        description = await generate_table_description(ddl, samples)
        print(f"      📝 Descrizione: {description[:80]}...") 
        
        # 4. SALVATAGGIO
        collection.add(
            documents=[description],
            metadatas=[{
                "table_name": table, 
                "original_ddl": ddl,
                "sample_data_snapshot": samples[:200] # Opzionale: salviamo anche un pezzetto di dati per debug
            }],
            ids=[table]
        )

    print(f"\n✅ Ingestion Completata! {collection.count()} tabelle.")

if __name__ == "__main__":
    asyncio.run(main())