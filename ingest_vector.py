import sqlite3
import json
import argparse
import asyncio
import os
from typing import List, Dict, Any
from src.embedding_factory import get_shared_embedding_function
import chromadb
from pydantic_ai import Agent
from src.embedding_factory import get_chroma_embedding_function
# Assumiamo che questi moduli esistano nel tuo progetto
from src.database import DatabaseManager
from src.config import get_model

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "cloneDefinitivoDB.db")
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_data")
COLLECTION_NAME = "langchain"

# --- PROMPT AVANZATO ---
DESCRIPTION_AGENT_PROMPT = """
Sei un esperto Data Steward e Database Administrator.
Il tuo compito è generare una documentazione semantica ricca per una tabella SQL, ottimizzata per la ricerca vettoriale (RAG).

Riceverai:
1. Il DDL della tabella (Create Table).
2. Un'analisi statistica dei dati (campioni e valori categorici rilevati).

Devi produrre una descrizione discorsiva in ITALIANO che spieghi:
1. **L'Entità Principale**: Cosa rappresenta la tabella nel mondo reale (es. "Ordini clienti", "Prodotti a magazzino").
2. **Le Colonne Chiave**: Descrivi le colonne basandoti sui dati forniti.
3. **Vocabolario Specifico**: Se l'analisi mostra valori categorici (es. status = 'shipped', 'pending'), ELENCALI ESPLICITAMENTE. Questo è fondamentale per permettere al sistema di mappare le domande dell'utente sui valori corretti.
4. **Relazioni**: Se intuisci chiavi esterne (es. `client_id`), menziona che la tabella collega questa entità ai clienti.

OUTPUT RICHIESTO: Solamente il testo della descrizione, senza preamboli o markdown extra.
"""

def analyze_columns_heuristics(db_path: str, table_name: str) -> str:
    """
    Esegue l'analisi 'intelligente' CON FILTRO NULL:
    - Controlla se le colonne sono vuote.
    - Se popolate, cerca categorie.
    - Prende campioni solo dalle colonne utili.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    
    analysis_report = []
    ignored_columns = [] # Lista per le colonne tutti NULL
    active_columns = []  # Lista per le colonne con dati
    
    try:
        # 1. Ottieni info sulle colonne
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        categorical_hints = []
        
        for col in columns:
            col_name = col['name']
            col_type = col['type'].upper()
            
            # --- CHECK NULL RAPIDO ---
            # Cerchiamo se esiste almeno 1 riga non NULL.
            # "SELECT 1" è molto più veloce di "COUNT(*)"
            cursor.execute(f"SELECT 1 FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 1")
            is_populated = cursor.fetchone()
            
            if not is_populated:
                ignored_columns.append(col_name)
                continue # Salta al prossimo ciclo, questa colonna è inutile
            
            active_columns.append(col_name)

            # --- CHECK CATEGORIE (Solo su colonne attive) ---
            if "CHAR" in col_type or "TEXT" in col_type or col_type == "":
                try:
                    cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 26")
                    values = [str(row[0]) for row in cursor.fetchall()]
                    
                    if 0 < len(values) <= 25:
                        vals_str = ", ".join(values)
                        categorical_hints.append(f"- Colonna '{col_name}': [{vals_str}]")
                except:
                    continue

        # --- COSTRUZIONE REPORT ---
        
        # A. Avviso colonne vuote (Utile per l'LLM per sapere cosa ignorare)
        if ignored_columns:
            analysis_report.append(f"⚠️ COLONNE COMPLETAMENTE VUOTE (IGNORATE): {', '.join(ignored_columns)}")
            analysis_report.append("")

        # B. Suggerimenti Categorie
        if categorical_hints:
            analysis_report.append("--- VALORI CATEGORICI RILEVATI ---")
            analysis_report.extend(categorical_hints)
            analysis_report.append("")

        # C. Campione Dati (Solo colonne attive)
        # Costruiamo la query solo con le colonne attive per risparmiare token
        if active_columns:
            cols_query = ", ".join(active_columns)
            cursor.execute(f"SELECT {cols_query} FROM {table_name} LIMIT 3")
            rows = cursor.fetchall()
            
            if rows:
                analysis_report.append("--- CAMPIONE DATI (Prime 3 righe, solo colonne attive) ---")
                analysis_report.append(f"Colonne visibili: {cols_query}")
                for row in rows:
                    # Convertiamo in dict per leggibilità, row_factory aiuta qui
                    analysis_report.append(str(dict(row)))
        else:
            analysis_report.append("Nessuna colonna attiva trovata (Tabella vuota?).")

    except Exception as e:
        return f"Errore durante l'analisi: {e}"
    finally:
        conn.close()
    
    return "\n".join(analysis_report)

async def process_single_table(db_manager, db_path, table_name, collection, agent):
    """
    Processa una singola tabella: Estrazione -> Generazione -> Salvataggio
    """
    print(f"   ⏳ Analisi tabella: {table_name}...")
    
    try:
        # A. Estrazione Dati Tecnici
        ddl = db_manager.get_table_ddl(table_name)
        
        # B. Analisi Euristica (Il 'Trick' dei valori categorici)
        # Nota: Eseguiamo codice sincrono in thread separato per non bloccare asyncio
        stats_text = await asyncio.to_thread(analyze_columns_heuristics, db_path, table_name)
        
        # C. Generazione Descrizione con LLM
        user_content = f"--- DDL TABELLA ---\n{ddl}\n\n{stats_text}"
        result = await agent.run(user_content)
        description = result.output # In PydanticAI v0.27+ usa .data, altrimenti .output
        
        # D. Preparazione Metadata per tools.py
        metadata_payload = {
            "table_name": table_name,
            "original_ddl": ddl,
            "generated_description": description,
            # Salviamo anche i hint categorici nel JSON per debug o uso futuro
            "categorical_hints": stats_text 
        }

        # E. Inserimento nel Vector DB
        # Creiamo un "Documento Ricco" che contiene sia la semantica che la tecnica
        # Questo assicura che il retrieval funzioni sia per "Fatturato" che per "imp_net_tot"
        rich_document = f"""
        DESCRIZIONE SEMANTICA:
        {description}
        
        DETTAGLI TECNICI E CATEGORIE:
        {stats_text}
        
        SCHEMA SQL (DDL):
        {ddl}
        """

        collection.add(
            documents=[rich_document], # <--- Vettorializziamo TUTTO
            metadatas=[{
                "table_name": table_name,
                "table_schema": json.dumps(metadata_payload)
            }],
            ids=[table_name]
        )
        print(f"   ✅ Tabella '{table_name}' completata.")
        return True

    except Exception as e:
        print(f"   ❌ ERRORE su tabella '{table_name}': {e}")
        return False

async def main():
    # --- 1. Setup ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=os.getenv("DB_PATH", DEFAULT_DB_PATH))
    parser.add_argument("--chroma_path", default=os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH))
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"❌ Database non trovato: {args.db_path}")
        return

    print("🔌 Connessione ai sistemi...")
    db_manager = DatabaseManager(args.db_path)
    
    # Setup Chroma
    print("🧠 Caricamento Modello Enterprise (BGE-M3)...")
    # Usa BGE-M3 per sfruttare la context window ampia (8k token) e non troncare la tua analisi euristica
    emb_fn = get_chroma_embedding_function()  # ✅ GIUSTO (Usa l'adapter) 

    chroma_client = chromadb.PersistentClient(path=args.chroma_path)

    # Reset pulito
    try: chroma_client.delete_collection(COLLECTION_NAME)
    except: pass
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=emb_fn
    )

    # Setup Agente (Unica istanza riutilizzata)
    agent = Agent(model=get_model(), system_prompt=DESCRIPTION_AGENT_PROMPT)

    # --- 2. Discovery ---
    tables = db_manager.search_tables(None)
    print(f"🚀 Trovate {len(tables)} tabelle. Inizio arricchimento parallelo...")

    

    # --- 3. Esecuzione Parallela ---
    # Creiamo un task per ogni tabella
    tasks = [
        process_single_table(db_manager, args.db_path, table, collection, agent)
        for table in tables
    ]
    
    # Eseguiamo tutto insieme
    results = await asyncio.gather(*tasks)
    
    success_count = sum(results)
    print(f"\n🏁 Finito! {success_count}/{len(tables)} tabelle indicizzate correttamente.")

if __name__ == "__main__":
    asyncio.run(main())