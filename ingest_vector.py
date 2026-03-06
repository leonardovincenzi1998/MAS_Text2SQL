import pickle
import re
import sqlite3
import json
import argparse
import asyncio
import os
from typing import List, Dict, Any, Tuple, Set
import chromadb
import hashlib

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from src.config import BM25_PATH

from src.embedding_factory import get_chroma_embedding_function
from src.database import DatabaseManager
from src.utils import get_canonical_name

from src.config import (
    DEFAULT_DB_PATH, CHROMA_PATH, COLLECTION_NAME, VALUE_COLLECTION_NAME,
    LLM_MODEL_NAME, BASE_URL, API_KEY
)
from src.prompts import DESCRIPTION_AGENT_PROMPT

# parses raw DDL to find foreign key references missed by metadata
def extract_implicit_fks_from_ddl(ddl: str) -> Set[str]:
    if not ddl:
        return set()
    
    # robust regex to capture REFERENCES table_name (column)
    pattern = r"""
        REFERENCES\s+
        (
            (?:
                "(?:[^"]+)" |
                `(?:[^`]+)` |
                \[(?:[^\]]+)\] |
                \w+
            )
        )
    """
    matches = re.findall(pattern, ddl, re.IGNORECASE | re.VERBOSE)
    
    normalized = set()
    for m in matches:
        clean_name = m.replace('"', '').replace('`', '').replace('[', '').replace(']', '')
        if '.' in clean_name:
            clean_name = clean_name.split('.')[-1]
        
        canon = get_canonical_name(clean_name)
        if canon:
            normalized.add(canon)
            
    return normalized

def analyze_columns_smart(
    db_path: str, 
    table_name: str, 
    threshold_sparsity: float = 0.95, 
    threshold_cardinality: int = 1
) -> Tuple[str, List[str], Dict[str, List[str]]]:
    # intelligent column analysis to identify significant columns for the LLM description
    # protects PKs/FKs, discards highly sparse or constant columns
    # detects categorical features (ratio check) to avoid listing unique IDs
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    report_lines: List[str] = []
    kept_columns: List[str] = []
    dropped_columns: List[str] = []
    categorical_hints: List[str] = []
    column_samples: Dict[str, List[str]] = {}
    
    try:
        # 1. identify PK and FK sets
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols_info = cursor.fetchall()
        pk_list: Set[str] = {row['name'] for row in cols_info if row['pk'] > 0}
        
        safe_table = table_name.replace('"', '""')
        cursor.execute(f'PRAGMA foreign_key_list("{safe_table}")')
        fk_list: Set[str] = {row['from'] for row in cursor.fetchall()}
        
        structural_cols = pk_list.union(fk_list)

        # 2. column analysis
        for col in cols_info:
            col_name = col['name']
            col_type = col['type'].upper()
            safe_col = f'"{col_name}"'
            
            is_structural = col_name in structural_cols
            is_boolean_or_important = (
                col_name.startswith("Is") or col_name.startswith("is") or 
                "BIT" in col_type or "BOOL" in col_type or
                "PERCENTUALE" in col_name.upper() or "VALORE" in col_name.upper()
)
            
            # formulate query based on numeric vs text types
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
            
            # drop or keep logic
            keep = False
            if total == 0:
                keep = False
            elif is_structural or is_boolean_or_important:
                keep = True
            elif sparsity >= threshold_sparsity:
                keep = False
            elif distinct <= threshold_cardinality:
                keep = False
            else:
                keep = True
            
            if keep:
                kept_columns.append(col_name)
                
                # new category logic via ratio check
                is_text = ("CHAR" in col_type or "TEXT" in col_type)
                not_pk = col_name not in pk_list
                few_distinct = 0 < distinct <= 25
                
                is_true_category = distinct <= 5 or (distinct < total * 0.5)

                if is_text and not_pk and few_distinct and is_true_category:
                    try:
                        cursor.execute(f'SELECT DISTINCT {safe_col} FROM "{safe_table}" WHERE {safe_col} IS NOT NULL LIMIT 25')
                        vals = [str(r[0]) for r in cursor.fetchall() if r[0] is not None and str(r[0]).strip() != '']
                        if vals:
                            vals_str = ", ".join(vals)
                            categorical_hints.append(f"- Colonna '{col_name}' ({len(vals)} val): [{vals_str}]")
                    except Exception:
                        pass
            else:
                if not is_structural:
                    dropped_columns.append(col_name)

        # 3. report construction
        if categorical_hints:
            report_lines.append("---CATEGORICAL VALUES FOUND ---")
            report_lines.extend(categorical_hints)
            report_lines.append("")

        if dropped_columns:
            drop_str = ", ".join(dropped_columns[:20]) 
            if len(dropped_columns) > 20: 
                drop_str += "..."
            report_lines.append(f"⚠️ COLUMNS IGNORED (Empty/Constant): {drop_str}\n")

        if kept_columns:
            cols_query = ", ".join([f'"{c}"' for c in kept_columns])
            cursor.execute(f'SELECT {cols_query} FROM "{safe_table}" LIMIT 20')
            rows = cursor.fetchall()
            
            for col in kept_columns:
                samples = set()
                for row in rows:
                    val = row[col]
                    # discard nulls and limit length to 40 chars to save tokens
                    if val is not None and str(val).strip() != '':
                        samples.add(str(val)[:40]) 
                
                if samples:
                    column_samples[col] = list(samples)[:3] 

            report_lines.append("--- DATA SAMPLES ---")
            for row in rows[:3]:
                report_lines.append(str(dict(row)))
        else:
            report_lines.append("No significant columns found.")

    except Exception as e:
        print(f"   ⚠️ ERRORE INTERNO in analyze_columns_smart su {table_name}: {e}")
        return f"Smart Analysis Error: {e}", [], {}
    finally:
        conn.close()
    
    return "\n".join(report_lines), kept_columns, column_samples


def get_foreign_keys_robust(db_path: str, table_name: str) -> List[Dict[str, str]]:
    # retrieves foreign keys using SQLite PRAGMA function
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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

def get_global_pks(db_path: str, tables: List[str]) -> Dict[str, str]:
    """Looks across all tables and maps Primary Key Column Names -> Table Names"""
    global_pks = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for t in tables:
        safe_table = t.replace('"', '""')
        cursor.execute(f'PRAGMA table_info("{safe_table}")')
        for row in cursor.fetchall():
            if row[5] > 0:  # row[5] è il flag 'pk'
                global_pks[row[1]] = t  # row[1] è il nome della colonna
    conn.close()
    return global_pks


async def process_single_table(
    db_manager: Any, 
    db_path: str, 
    table_name: str, 
    collection: Any, 
    value_collection: Any,
    agent: Agent, 
    semaphore: asyncio.Semaphore,
    global_pks: Dict[str, str]
) -> bool:
    # processes a single table pipeline: DDL Extraction -> Smart Profiling -> LLM Description -> ChromaDB Upsert
    print(f"   ⏳ Analisi tabella: {table_name}...")
    
    try:
        async with semaphore:
            real_table_name = table_name
            canonical_name = get_canonical_name(real_table_name)
            
            # --- PHASE A: FK EXTENSIONS (Explicit from DB + Implicit from DDL) ---
            foreign_keys = get_foreign_keys_robust(db_path, real_table_name)
            ddl = db_manager.get_table_ddl(real_table_name)

            implicit_fks = extract_implicit_fks_from_ddl(ddl)
            existing_targets = {fk.get("to_table_canonical") for fk in foreign_keys if fk.get("to_table_canonical")}
            
            for ref in implicit_fks:
                if ref not in existing_targets:
                    foreign_keys.append({
                        "to_table_canonical": ref,
                        "to_table_real": ref,  
                        "from_column": "inferita_da_ddl",
                        "to_column": "id"
                    })
                    existing_targets.add(ref)

            # --- PHASE A.2: HEURISTIC FK INFERENCE (Basato sul Naming) ---
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(f'PRAGMA table_info("{real_table_name}")')
            cols_info = cursor.fetchall()
            conn.close()

            for col in cols_info:
                col_name = col[1]
                is_pk = col[5] > 0
                if not is_pk and col_name in global_pks:
                    target_table = global_pks[col_name]
                    target_canon = get_canonical_name(target_table)
                    
                    if target_canon not in existing_targets:
                        foreign_keys.append({
                            "to_table_canonical": target_canon,
                            "to_table_real": target_table,
                            "from_column": col_name,
                            "to_column": col_name
                        })
                        existing_targets.add(target_canon)

            # --- PHASE B: SMART DATA PROFILING AND FALLBACK COLUMNS ---
            stats_text, significant_cols, column_samples = await asyncio.to_thread(
                analyze_columns_smart, db_path, real_table_name
            )

            # Fallback if there are no extracted columns (logic moved from tools.py)
            if not significant_cols:
                significant_cols = re.findall(r'(\w+)\s+(?:INT|TEXT|REAL|CHAR|DATE|FLOAT|DOUBLE|NUMERIC)', ddl, re.IGNORECASE)

            # Cleaning categories to limit payload (logic moved from tools.py)
            categorical_lines = []
            for line in stats_text.split('\n'):
                if line.strip().startswith("- Colonna"):
                    if len(line) > 150:
                        line = line[:145] + "...]"
                    categorical_lines.append(line.strip())
                    
            smart_hints = "\n".join(categorical_lines[:6])
            if len(categorical_lines) > 6: 
                smart_hints += "\n..."

            # --- PHASE C: LLM DESCRIPTION ---
            user_content = f"--- DDL TABLE ---\n{ddl}\n\n{stats_text}"
            result = await agent.run(user_content)
            description = getattr(result, "data", getattr(result, "output", str(result))).strip()
        
            # --- PHASE D: CREATION OF THE SLIM SCHEMA (No DDL, No Raw Profile) ---
            metadata_payload = {
                "table_name": real_table_name,
                "description": description,
                "columns": significant_cols,
                "categorical_values": smart_hints,
                "foreign_keys": foreign_keys,
                "column_samples": column_samples
            }

            # --- PHASE E: VECTOR DB UPSERT (The DDL goes only in the document, not in the metadata) ---
            vector_hints_lines = [line for line in stats_text.split('\n') if line.strip().startswith("- Colonna")]
            vector_hints = "\n".join(vector_hints_lines)

            rich_document = f"""
            TABLE NAME: {canonical_name} ({real_table_name})

            SEMANTIC DESCRIPTION:
            {description}

            MAIN COLUMNS CONTAINED:
            {', '.join(significant_cols)}

            CATEGORICAL VALUES FOUND:
            {vector_hints}
            """

            collection.upsert(
                documents=[rich_document],
                metadatas=[{
                    "canonical_name": canonical_name,
                    "table_schema": json.dumps(metadata_payload)
                }],
                ids=[canonical_name]
            )

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # take the column types to understand which ones are free text
                cursor.execute(f"PRAGMA table_info({real_table_name})")
                cols_info = cursor.fetchall()
                # filter only text/char columns that have been deemed significant
                text_cols = [c[1] for c in cols_info if c[1] in significant_cols and ("CHAR" in c[2].upper() or "TEXT" in c[2].upper())]
                
                val_docs = []
                val_metadatas = []
                val_ids = []
                
                for col in text_cols:
                    safe_col = f'"{col}"'
                    safe_table = f'"{real_table_name}"'
                    
                    cursor.execute(f"SELECT DISTINCT {safe_col} FROM {safe_table} WHERE {safe_col} IS NOT NULL")
                    rows = cursor.fetchall()
                    
                    for r in rows:
                        val = str(r[0]).strip()
                        
                        if len(val) >= 3:
                            val_docs.append(val)
                            val_metadatas.append({"table_name": real_table_name, "column_name": col})
                            
                            md5_hash = hashlib.md5(val.encode('utf-8')).hexdigest()
                            val_ids.append(f"{real_table_name}_{col}_{md5_hash}")
                
                conn.close()
                
                # block upsert to comply with ChromaDB limits
                if val_docs:
                    batch_size = 1000
                    for i in range(0, len(val_docs), batch_size):
                        value_collection.upsert(
                            documents=val_docs[i:i+batch_size],
                            metadatas=val_metadatas[i:i+batch_size],
                            ids=val_ids[i:i+batch_size]
                        )
                    print(f"   🔤 Inseriti {len(val_docs)} valori testuali per '{table_name}'.")
                    
            except Exception as e:
                print(f"   ⚠️ Errore nell'estrazione dei valori per '{table_name}': {e}")

            print(f"   ✅ Tabella '{table_name}' completata.")
        return True

    except Exception as e:
        print(f"   ❌ ERRORE su tabella '{table_name}': {e}")
    return False

def get_model():
    # configures the model by setting environment variables
    # this method is safe because it bypasses syntax differences between library versions
    
    # 1. set environment variables that the internal 'openai' library listens to
    os.environ['OPENAI_BASE_URL'] = BASE_URL
    os.environ['OPENAI_API_KEY'] = API_KEY

    # 2. initialize the model passing ONLY the name
    return OpenAIChatModel(model_name=LLM_MODEL_NAME)


async def main():
    # argument setup
    parser = argparse.ArgumentParser(description="Ingest DB schema into Chroma Vector Database")
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH)
    parser.add_argument("--chroma_path", default=CHROMA_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"❌ Database non trovato: {args.db_path}")
        return

    print("🔌 Connessione ai sistemi...")
    db_manager = DatabaseManager(args.db_path)
    
    # chromadb setup
    print("🧠 Caricamento Modello Enterprise (BGE-M3)...")
    emb_fn = get_chroma_embedding_function()

    chroma_client = chromadb.PersistentClient(path=args.chroma_path)

    # optional clean reset
    try: 
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception: 
        pass
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=emb_fn
    )

    try:
        chroma_client.delete_collection(VALUE_COLLECTION_NAME)
    except Exception:
        pass
    value_collection = chroma_client.get_or_create_collection(
        name=VALUE_COLLECTION_NAME, embedding_function=emb_fn
    )
    
    agent = Agent(model=get_model(), system_prompt=DESCRIPTION_AGENT_PROMPT)

    # discovery and execution
    tables = db_manager.search_tables(None)
    print(f"🚀 Trovate {len(tables)} tabelle. Inizio arricchimento parallelo...")

    global_pks = get_global_pks(args.db_path, tables)

    max_conc = int(os.getenv("INGEST_MAX_CONCURRENCY", "4"))
    semaphore = asyncio.Semaphore(max(1, max_conc))

    tasks = [
        process_single_table(db_manager, args.db_path, table, collection, value_collection, agent, semaphore, global_pks)
        for table in tables
    ]
    
    results = await asyncio.gather(*tasks)
    
    success_count = sum(results)
    print(f"\n🏁 Finito! {success_count}/{len(tables)} tabelle indicizzate correttamente.")

    print("📚 Costruzione indice lessicale BM25 (Hybrid Retrieval)...")

    try:
        # Recover the entire vector database you just created
        all_data = collection.get()
        docs_for_bm25 = []
        
        # Converts Chroma data into LangChain Document objects
        for doc_text, meta in zip(all_data['documents'], all_data['metadatas']):
            docs_for_bm25.append(Document(page_content=doc_text, metadata=meta))
            
        if docs_for_bm25:
            # Train the BM25 on documents
            bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
            
            # Save the index to disk so that it can be quickly loaded into the tools
            with open(BM25_PATH, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            print("✅ Indice BM25 completato e salvato su disco.")
        else:
            print("⚠️ Nessun documento trovato per l'indice BM25.")
            
    except Exception as e:
        print(f"❌ Errore durante la creazione dell'indice BM25: {e}")

if __name__ == "__main__":
    asyncio.run(main())