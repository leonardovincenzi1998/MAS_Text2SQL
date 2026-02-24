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
from src.naming import get_canonical_name

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

def analyze_columns_smart(db_path: str, table_name: str, threshold_sparsity=0.95, threshold_cardinality=1) -> tuple[str, list[str], dict]:
    """
    Analisi intelligente delle colonne per identificare quelle significative da includere nella descrizione:
    - Protegge PK/FK.
    - Scarta colonne sparse o costanti.
    - Rileva categorie (Ratio Check),
      evitando di listare nomi o codici univoci in tabelle piccole.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    report_lines = []
    kept_columns = []
    dropped_columns = []
    categorical_hints = []
    
    try:
        # 1. Identifica PK e FK
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols_info = cursor.fetchall()
        pk_list = {row['name'] for row in cols_info if row['pk'] > 0}
        
        safe_table = table_name.replace('"', '""')
        cursor.execute(f'PRAGMA foreign_key_list("{safe_table}")')
        fk_list = {row['from'] for row in cursor.fetchall()}
        
        structural_cols = pk_list.union(fk_list)

        # 2. Analisi Colonne
        for col in cols_info:
            col_name = col['name']
            col_type = col['type'].upper()
            safe_col = f'"{col_name}"'
            
            is_structural = col_name in structural_cols
            
            # Query Statistica
            if any(x in col_type for x in ['INT', 'REAL', 'NUM', 'DEC', 'FLOAT', 'DOUBLE']):
                empty_condition = f"{safe_col} IS NULL OR {safe_col} = 0"
            else:
                empty_condition = f"{safe_col} IS NULL OR {safe_col} = ''"

            stats_query = f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT {safe_col}) as distinct_count,
                    SUM(CASE WHEN {empty_condition} THEN 1 ELSE 0 END) as empty_count
                FROM "{safe_table}"
            """
            
            cursor.execute(stats_query)
            row = cursor.fetchone()
            
            total = row['total']
            distinct = row['distinct_count']
            empty = row['empty_count'] or 0
            
            sparsity = (empty / total) if total > 0 else 1.0
            
            # Logica DROP/KEEP
            keep = False
            if total == 0:
                keep = False
            elif is_structural:
                keep = True
            elif sparsity >= threshold_sparsity:
                keep = False
            elif distinct <= threshold_cardinality:
                keep = False
            else:
                keep = True
            
            if keep:
                kept_columns.append(col_name)
                
                # --- NUOVA LOGICA CATEGORIE (Ratio Check) ---
                is_text = ("CHAR" in col_type or "TEXT" in col_type)
                not_pk = col_name not in pk_list
                few_distinct = 0 < distinct <= 25
                
                # Regola: 
                # O sono pochissimi in assoluto (<= 5) -> Es. flag booleani
                # O sono pochi rispetto al totale delle righe (< 50%) -> Es. Categorie vere
                is_true_category = distinct <= 5 or (distinct < total * 0.5)

                if is_text and not_pk and few_distinct and is_true_category:
                    try:
                        cursor.execute(f'SELECT DISTINCT {safe_col} FROM "{safe_table}" WHERE {safe_col} IS NOT NULL LIMIT 25')
                        vals = [str(r[0]) for r in cursor.fetchall() if r[0] is not None and str(r[0]).strip() != '']
                        if vals:
                            vals_str = ", ".join(vals)
                            categorical_hints.append(f"- Colonna '{col_name}' ({len(vals)} val): [{vals_str}]")
                    except:
                        pass
            else:
                if not is_structural:
                    dropped_columns.append(col_name)

        # 3. Costruzione Report
        if categorical_hints:
            report_lines.append("--- VALORI CATEGORICI RILEVATI ---")
            report_lines.extend(categorical_hints)
            report_lines.append("")

        if dropped_columns:
            drop_str = ", ".join(dropped_columns[:20]) 
            if len(dropped_columns) > 20: drop_str += "..."
            report_lines.append(f"⚠️ COLONNE IGNORATE (Vuote/Costanti): {drop_str}")
            report_lines.append("")

        column_samples = {}
        
        if kept_columns:
            # Estraiamo i sample da passare strutturati al RAG
            cols_query = ", ".join([f'"{c}"' for c in kept_columns])
            cursor.execute(f'SELECT {cols_query} FROM "{safe_table}" LIMIT 20')
            rows = cursor.fetchall()
            
            for col in kept_columns:
                samples = set()
                for row in rows:
                    val = row[col]
                    # Scartiamo i null e limitiamo la lunghezza a 40 char per non esplodere i token
                    if val is not None and str(val).strip() != '':
                        samples.add(str(val)[:40]) 
                
                if samples:
                    column_samples[col] = list(samples)[:3] # Prendiamo massimo 3 valori diversi

            # Aggiungiamo anche le righe grezze al report per l'Agent 1 (opzionale)
            report_lines.append("--- CAMPIONE DATI ---")
            for row in rows[:3]:
                report_lines.append(str(dict(row)))
        else:
            report_lines.append("Nessuna colonna significativa trovata.")

    except Exception as e:
        return f"Errore analisi smart: {e}", [], {}
    finally:
        conn.close()
    
    return "\n".join(report_lines), kept_columns, column_samples

def get_foreign_keys_robust(db_path: str, table_name: str) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # quoting per SQLite
    safe_table = table_name.replace('"', '""')
    cursor.execute(f'PRAGMA foreign_key_list("{safe_table}")')
    fks = cursor.fetchall()
    conn.close()

    results = []
    for fk in fks:
        referenced_table_real = fk[2]
        results.append({
            "from_column": fk[3],
            "to_table_canonical": get_canonical_name(referenced_table_real),
            "to_table_real": referenced_table_real,
            "to_column": fk[4]
        })
    return results


async def process_single_table(db_manager, db_path, table_name, collection, agent, semaphore):
    """
    Processa una singola tabella: Estrazione -> Generazione -> Salvataggio
    """
    print(f"   ⏳ Analisi tabella: {table_name}...")
    
    try:
        async with semaphore:
            real_table_name = table_name
            canonical_name = get_canonical_name(real_table_name)
            
            # Estrazione FK e DDL
            foreign_keys = get_foreign_keys_robust(db_path, real_table_name)
            ddl = db_manager.get_table_ddl(real_table_name)

            # B. Analisi SMART (Sostituisce quella vecchia euristica)
            # Rileva PK/FK, pulisce colonne vuote/costanti e genera statistiche
            stats_text, significant_cols, column_samples = await asyncio.to_thread(analyze_columns_smart, db_path, real_table_name)

            # C. Generazione Descrizione con LLM
            user_content = f"--- DDL TABELLA ---\n{ddl}\n\n{stats_text}"
            result = await agent.run(user_content)
            description = result.output # Nota: Adattato a result.data come da PydanticAI recente (o result.output a seconda della versione)
        
            # D. Preparazione Metadata per tools.py
            metadata_payload = {
                "real_table_name": real_table_name,  # nuovo standard
                "significant_cols": significant_cols,
                "canonical_name": canonical_name,
                "foreign_keys": foreign_keys,
                "original_ddl": ddl,
                "generated_description": description,
                "data_profile": stats_text,           # Qui c'è il report 'Smart'
                "column_samples": column_samples      # Campioni per ogni colonna significativa
            }

            # E. Inserimento nel Vector DB
            rich_document = f"""
            DESCRIZIONE SEMANTICA:
            {description}
            
            DETTAGLI TECNICI E CATEGORIE:
            {stats_text}
            
            SCHEMA SQL (DDL):
            {ddl}
            """

            collection.upsert(
                documents=[rich_document],
                metadatas=[{
                    #"table_name": real_table_name,      # compatibilità
                    #"real_table_name": real_table_name,
                    "canonical_name": canonical_name,
                    "table_schema": json.dumps(metadata_payload)
                }],
                ids=[canonical_name]
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
    emb_fn = get_chroma_embedding_function()

    chroma_client = chromadb.PersistentClient(path=args.chroma_path)

    # Reset pulito (opzionale: commenta se vuoi fare upsert incrementale)
    try: chroma_client.delete_collection(COLLECTION_NAME)
    except: pass
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=emb_fn
    )

    # Setup Agente
    agent = Agent(model=get_model(), system_prompt=DESCRIPTION_AGENT_PROMPT)

    # --- 2. Discovery ---
    tables = db_manager.search_tables(None)
    print(f"🚀 Trovate {len(tables)} tabelle. Inizio arricchimento parallelo...")

    max_conc = int(os.getenv("INGEST_MAX_CONCURRENCY", "4"))
    semaphore = asyncio.Semaphore(max(1, max_conc))

    # --- 3. Esecuzione Parallela ---
    tasks = [
        process_single_table(db_manager, args.db_path, table, collection, agent, semaphore)
        for table in tables
    ]
    
    results = await asyncio.gather(*tasks)
    
    success_count = sum(results)
    print(f"\n🏁 Finito! {success_count}/{len(tables)} tabelle indicizzate correttamente.")

if __name__ == "__main__":
    asyncio.run(main())