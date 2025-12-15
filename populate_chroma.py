import os
import shutil
import sqlite3
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- CONFIGURAZIONE PERCORSI ---
# Calcola i percorsi in modo dinamico come abbiamo fatto per tools.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(CURRENT_DIR, "chroma_db_data")
SQLITE_DB_PATH = os.path.join(CURRENT_DIR, "Terreni_Fabbricati.db") # Assicurati che il nome sia esatto!

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_schema_representation(db_path):
    """Estrae lo schema dal DB SQLite e crea descrizioni testuali per l'embedding."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"❌ Impossibile trovare il database SQL in: {db_path}")

    print(f"📖 Leggendo schema da: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Prende tutte le tabelle
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
    
    documents = []
    
    for table in tables:
        # Prende le colonne
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_desc = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
        
        # Prende le Foreign Keys (importante per le relazioni!)
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        fk_desc = ""
        if fks:
            fk_list = [f"linked to {fk[2]}.{fk[3]}" for fk in fks]
            fk_desc = f". Relazioni: {', '.join(fk_list)}"

        # --- CREAZIONE DEL CONTENUTO SEMANTICO ---
        # Questo è il testo che l'AI userà per cercare. Deve essere descrittivo.
        # Mischiamo Italiano (per i nomi colonne) e Inglese (per struttura) per robustezza.
        text_content = (
            f"Tabella: {table}. "
            f"Colonne: {col_desc}. "
            f"Contesto: Questa tabella contiene dati riguardanti {table}{fk_desc}."
        )
        
        # Creiamo il JSON completo da salvare nei metadati (per l'Agente 2)
        metadata_schema = {
            "table_name": table,
            "columns": [col[1] for col in columns],
            "ddl": f"CREATE TABLE {table} ({col_desc})" # Semplificato
        }

        # Creiamo il documento LangChain
        doc = Document(
            page_content=text_content,
            metadata={
                "table_name": table,
                "table_schema": json.dumps(metadata_schema) # Salviamo come stringa JSON
            }
        )
        documents.append(doc)
        print(f"   👉 Preparata tabella: {table}")

    conn.close()
    return documents

def main():
    print("="*50)
    print("🚀 INIZIO POPOLAMENTO CHROMADB")
    print("="*50)

    # 1. PULIZIA: Rimuovi il vecchio DB se esiste
    if os.path.exists(CHROMA_PATH):
        print(f"🗑️  Rimozione vecchia cartella database: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    else:
        print("✨ Creazione nuovo database da zero.")

    # 2. CARICAMENTO DATI
    try:
        docs = get_schema_representation(SQLITE_DB_PATH)
    except Exception as e:
        print(e)
        return

    # 3. CREAZIONE VETTORI E SALVATAGGIO
    print(f"\n🧠 Caricamento modello di embedding '{EMBEDDING_MODEL}'...")
    embedding_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"💾 Indicizzazione di {len(docs)} tabelle in ChromaDB...")
    
    # Questo comando crea il DB, inserisce i dati e SALVA su disco
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_func,
        persist_directory=CHROMA_PATH
    )

    # 4. VERIFICA FINALE
    count = vectorstore._collection.count()
    print("\n" + "="*50)
    if count > 0:
        print(f"✅ SUCCESSO! Database popolato con {count} tabelle.")
        print(f"📂 Posizione: {CHROMA_PATH}")
    else:
        print("❌ FALLIMENTO: Il database è stato creato ma risulta VUOTO.")
    print("="*50)

if __name__ == "__main__":
    main()