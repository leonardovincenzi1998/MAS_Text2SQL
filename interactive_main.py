import argparse
import asyncio
import os
import sys
import warnings
import traceback
import sqlglot
import sqlite3
import re
from sqlglot.expressions import Table, Column
from langchain_core.messages import HumanMessage
from src.config import DEFAULT_DB_PATH

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

# METRICS FUNCTIONS (From batch_evaluator2.py)

def extract_tables_from_sql(sql: str) -> set:
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        return set(t.name.lower() for t in parsed.find_all(Table))
    except Exception:
        sql_clean = sql.replace('\n', ' ')
        tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)', sql_clean, re.IGNORECASE)
        return set([t.lower() for t in tables])

def compare_execution_results(db_path: str, golden_sql: str, generated_sql: str) -> bool:
    if not generated_sql or not golden_sql:
        return False
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(golden_sql)
        golden_res = cursor.fetchall()
        cursor.execute(generated_sql)
        generated_res = cursor.fetchall()
        if "ORDER BY" in golden_sql.upper():
            return golden_res == generated_res
        else:
            return sorted(golden_res) == sorted(generated_res)
    except Exception:
        return False
    finally:
        if conn:
            conn.close()

def extract_columns_from_sql(sql: str) -> set:
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        alias_map = {}
        table_names = []
        for t in parsed.find_all(Table):
            t_name = t.name.lower()
            table_names.append(t_name)
            alias_map[t_name] = t_name 
            if t.alias:
                alias_map[t.alias.lower()] = t_name
                
        columns = set()
        for c in parsed.find_all(Column):
            col_name = c.name.lower()
            if c.table:
                real_table = alias_map.get(c.table.lower(), c.table.lower())
                columns.add(f"{real_table}.{col_name}")
            else:
                if len(set(table_names)) == 1:
                    columns.add(f"{table_names[0]}.{col_name}")
                else:
                    columns.add(col_name)
        return columns
    except Exception:
        sql_no_strings = re.sub(r"'.*?'", "", sql.lower())
        matches = re.findall(r'\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b', sql_no_strings)
        return set(f"{t}.{c}" for t, c in matches if t not in ("count", "sum", "avg", "max", "min"))

# interactive main
async def main():
    parser = argparse.ArgumentParser(description="Run the Text-to-SQL agent in INTERACTIVE mode")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
    parser.add_argument("--chroma_path", type=str, default=None, help="Path to the Chroma Vector DB directory")
    
    args = parser.parse_args()
    db_path = args.db
    chroma_path = args.chroma_path

    if chroma_path:
        os.environ["CHROMA_PATH"] = chroma_path

    from src.graph import app 

    if not os.path.exists(db_path):
        print(f"❌ ERRORE CRITICO: Il file database '{db_path}' non esiste.")
        print(f"Percorso assoluto cercato: {os.path.abspath(db_path)}")
        sys.exit(1)

    print("\n" + "="*60)
    print("🤖 AGENTE SQL ATTIVO - MODALITÀ INTERATTIVA")
    print(f"📁 DB Connesso: {db_path}")
    if chroma_path:
        print(f"📚 Vector DB: {chroma_path}")
    print("="*60)
    print("💡 Istruzioni: Scrivi la tua domanda e premi Invio.")
    print("   Scrivi 'exit', 'quit' o 'esci' per terminare.\n")

    while True:
        try:
            user_query = input("💬 Scrivi la tua domanda: ").strip()

            if user_query.lower() in ["exit", "quit", "esci"]:
                print("\n👋 Chiusura sessione. A presto!")
                break
            
            if not user_query:
                continue
                
            # Optional Golden SQL request for metrics
            print("   (Opzionale) Se vuoi calcolare le metriche, inserisci la Golden SQL.")
            print("   💡 Puoi incollare su più righe. La lettura si ferma quando trova un ';' oppure se premi Invio su una riga vuota.")
            print("🥇 Golden SQL (premi solo Invio per saltare): ")
            
            golden_sql_lines = []
            while True:
                line = input()
                # If press Enter straight away without typing anything, it skips the metrics
                if not line.strip() and len(golden_sql_lines) == 0:
                    break
                
                golden_sql_lines.append(line)
                
                # Stop reading if there is a semicolon, or if an empty line is sent after pasting
                if ';' in line or (not line.strip() and len(golden_sql_lines) > 0):
                    break
                    
            golden_sql = "\n".join(golden_sql_lines).strip()

            print(f"\n🤖 Domanda Utente: '{user_query}'\n")

            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "db_path": db_path,
                "selected_tables": [],
                "error": None,
                "retry_count": 0
            }

            final_state = await app.ainvoke(initial_state)

            print("-" * 50)
            
            if final_state.get("error"):
                print(f"❌ Errore durante l'esecuzione: {final_state['error']}")
            else:
                print("🚀 ESTRAZIONE E SELEZIONE COMPLETATA\n")

            extraction = final_state.get("extraction_result")
            if extraction:
                print(f"📋 [Agente 1] Intento:     {getattr(extraction, 'intent', 'N/A')}")
                print(f"🔑 [Agente 1] Entità:      {getattr(extraction, 'entities', [])}")
                print(f"🗂️ [Agente 1] Filtri:      {getattr(extraction, 'filters', [])}")
                print(f"🔎 [Agente 1] Search Keywords: {getattr(extraction, 'search_keywords', [])}")

            print("\n📚 [Vector DB] Schema Recuperato:")
            if final_state.get("parsed_schema"):
                schema_len = len(str(final_state['parsed_schema']))
                print(f"   (JSON Schema trovato, lunghezza stimata: {schema_len} caratteri)")
            else:
                print("   Nessuno schema trovato.")

            messages = final_state.get("messages", [])
            for msg in messages:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if "✅ Selected Tables:" in content:
                    print(f"\n🧠 [Ragionamento Agente 2 - Table Selector]:\n{content}")
                elif "✅ Selected Columns:" in content:
                    print(f"\n🧠 [Ragionamento Agente 2.5 - Column Selector]:\n{content}")

            generated_sql = final_state.get("generated_sql")
            if generated_sql:
                print(f"\n✍️  [Agente 3] SQL GENERATO:")
                print(generated_sql)

            # CALCULATION AND PRINTING OF SOTA DIMENSIONS                     
            execution_status = final_state.get("execution_status", False)
            print("\n📊 --- METRICHE DI ESECUZIONE ---")
            print(f"Syntax Accuracy (Eseguibile senza errori): {'✅ Passata' if execution_status else '❌ Fallita'}")
            
            if golden_sql and generated_sql:
                selected_tables = final_state.get("selected_tables", [])
                selected_cols_dict = final_state.get("selected_columns") or {}
                
                # Calculating EX-Match
                ex_match = False
                if execution_status:
                    ex_match = compare_execution_results(db_path, golden_sql, generated_sql)
                
                # Calculating Table Recall & Precision
                golden_tables = extract_tables_from_sql(golden_sql)
                golden_tables_lower = set([t.lower() for t in golden_tables])
                selected_tables_lower = set([t.lower() for t in selected_tables])
                
                tp_tab = len(golden_tables_lower.intersection(selected_tables_lower))
                table_recall = tp_tab / len(golden_tables) if golden_tables else 0.0
                table_precision = tp_tab / len(selected_tables_lower) if selected_tables_lower else 0.0
                
                # Calculating Column Recall & Precision
                selected_cols_fq = set()
                for t_name, cols in selected_cols_dict.items():
                    for col in cols:
                        selected_cols_fq.add(f"{t_name.lower()}.{col.lower()}")
                
                golden_cols_fq = extract_columns_from_sql(golden_sql)
                tp_col = len(golden_cols_fq.intersection(selected_cols_fq))
                col_recall = tp_col / len(golden_cols_fq) if golden_cols_fq else 0.0
                col_precision = tp_col / len(selected_cols_fq) if selected_cols_fq else 0.0
                
                # Screen output (will be automatically captured in the log file via ‘tee’ in bash)
                print(f"EX-Match (Accuratezza Logica):           {'✅ Passato (1.0)' if ex_match else '❌ Fallito (0.0)'}")
                print(f"Table Recall:                            {table_recall:.2f}")
                print(f"Table Precision:                         {table_precision:.2f}")
                print(f"Column Recall:                           {col_recall:.2f}")
                print(f"Column Precision:                        {col_precision:.2f}")
            elif not golden_sql:
                print("⚠️  Metriche SOTA (EX-Match, Recall, Precision) saltate: Golden SQL non fornita.")

            print("-" * 50)
            print("\n" + "x" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interruzione manuale rilevata. Uscita...")
            break
        except Exception as e:
            print(f"❌ Exception non gestita: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())