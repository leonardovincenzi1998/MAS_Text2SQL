import argparse
import asyncio
import os
import sqlglot
import json
import re
import warnings
import sqlite3
import gc
from sqlglot.expressions import Table, Column

# Disabilita i warning di pydantic per un log più pulito
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

from langchain_core.messages import HumanMessage
from src.config import DEFAULT_DB_PATH

def parse_golden_set(filepath: str) -> list:
    """
    Estrae le domande e le query SQL attese dal file di test.
    Cerca il pattern: numero. "Domanda" seguito dalla query SQL.
    """
    parsed_data = []
    if not os.path.exists(filepath):
        print(f"❌ File {filepath} non trovato.")
        return parsed_data

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'(\d+)\.\s+"([^"]+)"\s*\n(.*?)(?=\n\d+\.\s+"|\Z)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        parsed_data.append({
            "id": int(match.group(1)),
            "question": match.group(2).strip(),
            "golden_sql": match.group(3).strip()
        })
            
    return parsed_data

def extract_tables_from_sql(sql: str) -> set:
    """ Estrae in modo preciso i nomi delle tabelle usando l'AST di sqlglot. """
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        return set(t.name.lower() for t in parsed.find_all(Table))
    except Exception:
        # Fallback di sicurezza con Regex se il parser fallisce per sintassi strana
        sql_clean = sql.replace('\n', ' ')
        tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)', sql_clean, re.IGNORECASE)
        return set([t.lower() for t in tables])

def compare_execution_results(db_path: str, golden_sql: str, generated_sql: str) -> bool:
    """
    SOTA Metric (EX-Match): Esegue entrambe le query e confronta i set di risultati estratti.
    Se c'è ORDER BY controlla l'ordine, altrimenti usa l'equivalenza dei Set.
    """
    if not generated_sql or not golden_sql:
        return False
        
    conn = None
    try:
        # Usa modalità read-only per sicurezza
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        cursor.execute(golden_sql)
        golden_res = cursor.fetchall()
        
        cursor.execute(generated_sql)
        generated_res = cursor.fetchall()
        
        # Se c'è un ordine esplicito, le liste devono combaciare esattamente
        if "ORDER BY" in golden_sql.upper():
            return golden_res == generated_res
        else:
            return sorted(golden_res) == sorted(generated_res)
            
    except Exception as e:
        return False
    finally:
        if conn:
            conn.close()

def extract_columns_from_sql(sql: str) -> set:
    """ Estrae in modo preciso i veri nomi di colonna usando l'AST di sqlglot, 
        ignorando alias di tabella, alias di colonna e stringhe. """
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        # Estrae solo il nome effettivo della colonna (es. da "a.Descrizione" prende "descrizione")
        columns = set(c.name.lower() for c in parsed.find_all(Column))
        return columns
    except Exception:
        # Fallback con Regex (ora migliorata per ignorare le stringhe tra apici)
        sql_keywords = {"select", "from", "join", "where", "and", "or", "group", "by", 
                        "order", "having", "limit", "as", "on", "is", "null", "not", 
                        "in", "exists", "count", "sum", "avg", "max", "min", "distinct", 
                        "desc", "asc", "cast", "strftime", "lower", "upper"}
        
        # Rimuove le stringhe tra apici (es. 'RO')
        sql_no_strings = re.sub(r"'.*?'", "", sql.lower())
        # Rimuove prefissi di tabelle (es. bm.Valore -> Valore)
        clean_sql = re.sub(r'[a-zA-Z0-9_]+\.', '', sql_no_strings) 
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', clean_sql)
        return set([w for w in words if w not in sql_keywords])

async def main():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL agent in BATCH mode for Evaluation")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
    parser.add_argument("--golden_set", type=str, default="set_domande.txt", help="Path to the Golden Set txt file")
    parser.add_argument("--output", type=str, default="metriche_sota.json", help="Path for the output JSON results")
    args = parser.parse_args()

    # Importa il workflow solo dopo aver settato le variabili d'ambiente
    from src.graph import app 

    print("="*70)
    print("📊 AVVIO VALUTAZIONE BATCH - METRICHE SOTA")
    print("="*70)

    eval_dataset = parse_golden_set(args.golden_set)
    if not eval_dataset:
        return
        
    print(f"Trovate {len(eval_dataset)} domande nel file. Inizio elaborazione...\n")
    
    results = []

    for item in eval_dataset:
        q_id = item["id"]
        user_query = item["question"]
        golden_sql = item["golden_sql"]
        
        print(f"▶️ [{q_id}/{len(eval_dataset)}] '{user_query}'")
        
        # initial_state = {
        #     "messages": [HumanMessage(content=user_query)],
        #     "user_query": user_query,
        #     "db_path": args.db,
        #     "selected_tables": [],
        #     "error": None,
        #     "retry_count": 0
        # }

        max_domanda_retries = 2
        
        for attempt in range(max_domanda_retries):
            try:

                current_state = {
                        "messages": [HumanMessage(content=user_query)],
                        "user_query": user_query,
                        "db_path": args.db,
                        "selected_tables": [],
                        "error": None,
                        "retry_count": 0
                    }
                
                # Timeout globale di 4 minuti (240 secondi) per evitare blocchi infiniti
                final_state = await asyncio.wait_for(app.ainvoke(current_state), timeout=240.0)
                
                errore_fatale = final_state.get("error") or final_state.get("error_traceback")
                esecuzione_ok = final_state.get("execution_status")
                
                if errore_fatale and not esecuzione_ok and attempt < max_domanda_retries - 1:
                    print(f"   ⚠️ Errore critico rilevato: {errore_fatale}")
                    print(f"   🔄 Ritento l'intera domanda da capo (Tentativo {attempt + 2}/{max_domanda_retries})...")
                    continue

                generated_sql = final_state.get("generated_sql")
                execution_status = final_state.get("execution_status")

                print(f"   📝 GOLDEN SQL: {golden_sql.replace(chr(10), ' ')}") # chr(10) è \n
                if generated_sql:
                    print(f"   🚀 GEN SQL:    {generated_sql.replace(chr(10), ' ')}")
                else:
                    print(f"   🚀 GEN SQL:    Nessuna query generata!")
                    
                selected_tables = final_state.get("selected_tables", [])
                retry_count = final_state.get("retry_count", 0)
                selected_cols_dict = final_state.get("selected_columns") or {}
                
                # --- CALCOLO METRICHE SOTA ---
                
                # 1. Execution Match (EX-Match)
                ex_match = False
                if execution_status:
                    ex_match = compare_execution_results(args.db, golden_sql, generated_sql)
                
                # 2. Schema Linking Metrics (Table Selector)
                golden_tables = extract_tables_from_sql(golden_sql)
                golden_tables_lower = set([t.lower() for t in golden_tables])

                selected_tables_lower = set([t.lower() for t in selected_tables])
                
                true_positives = len(golden_tables_lower.intersection(selected_tables_lower))
                table_recall = true_positives / len(golden_tables) if golden_tables else 0.0
                table_precision = true_positives / len(selected_tables_lower) if selected_tables_lower else 0.0
                
                # 3. Schema Linking Metrics (Column Selector)
                # Appiattiamo tutte le colonne scelte in un set
                selected_cols_flat = set(col.lower() for cols in selected_cols_dict.values() for col in cols)
                golden_cols_raw = extract_columns_from_sql(golden_sql)
                
                # FIX: Rimuoviamo i nomi delle tabelle estratti per sbaglio dalla Golden
                golden_cols = golden_cols_raw - golden_tables_lower
                
                col_true_positives = len(golden_cols.intersection(selected_cols_flat))
                col_recall = col_true_positives / len(golden_cols) if golden_cols else 0.0
                col_precision = col_true_positives / len(selected_cols_flat) if selected_cols_flat else 0.0

                result_record = {
                    "id": q_id,
                    "question": user_query,
                    "golden_sql": golden_sql,
                    "generated_sql": generated_sql,
                    "execution_status": execution_status,
                    "ex_match": ex_match,
                    "table_recall": table_recall,
                    "table_precision": table_precision,
                    "selected_tables_agent2": list(selected_tables),
                    "col_recall": col_recall,
                    "col_precision": col_precision,
                    "retry_count": retry_count,
                    "error": final_state.get("error_traceback") or final_state.get("error")
                }
                results.append(result_record)
                
                print(f"\n   📊 [DETTAGLIO SCHEMA LINKING]")
                print(f"   📂 TABELLE:")
                print(f"      - Attese (Golden): {list(golden_tables)}")
                print(f"      - Scelte (Agent 2): {list(selected_tables)}")
                
                print(f"   🏷️  COLONNE:")
                print(f"      - Attese (Golden): {list(golden_cols)}")
                # Mostriamo le colonne raggruppate per tabella per leggibilità
                col_info = ", ".join([f"{t}: {cols}" for t, cols in selected_cols_dict.items()])
                print(f"      - Scelte (Agent 2.5): {col_info}")
                print(f"   --------------------------------------------------")
                
                status_icon = "🏆" if ex_match else ("✅" if execution_status else "❌")
                print(f"   {status_icon} Sintassi: {execution_status} | EX-Match: {ex_match}")
                print(f"   🔍 Table: Recall {table_recall*100:.0f}% - Precision {table_precision*100:.0f}%")
                print(f"   🎯 Column: Recall {col_recall*100:.0f}% - Precision {col_precision*100:.0f}%")
                
                if not ex_match and execution_status:
                    print("   ⚠️ L'SQL generato gira, ma estrae dati sbagliati (Allucinazione Logica/Topologica)!")
                    
                break

            except asyncio.TimeoutError:
                if attempt < max_domanda_retries - 1:
                    print(f"   ⏳ Timeout! Ritento l'intera domanda da capo (Tentativo {attempt + 2}/{max_domanda_retries})...")
                    continue
                else:
                    print(f"   ❌ Timeout definitivo dopo {max_domanda_retries} tentativi.")
                    results.append({"id": q_id, "ex_match": False, "execution_status": False, "error": "Timeout", "retry_count": 0})
                    break

            except Exception as e:
                if attempt < max_domanda_retries - 1:
                    print(f"   💥 Eccezione ({str(e)}). Ritento (Tentativo {attempt + 2}/{max_domanda_retries})...")
                    continue
                else:
                    print(f"   ❌ Eccezione definitiva: {e}")
                    results.append({"id": q_id, "ex_match": False, "execution_status": False, "error": str(e), "retry_count": 0})
                    break

        print("   🧘‍♂️ Pausa di 5 secondi per pulizia cache e raffreddamento GPU...")
        gc.collect() 
        await asyncio.sleep(5.0)

    # Salvataggio e Report Finale
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    valid_results = [r for r in results if r.get("error") != "Timeout"]
    
    # Calcolo Medie e Aggregazioni per la Dashboard
    ex_syntax_count = sum(1 for r in valid_results if r.get("execution_status") is True)
    ex_match_count = sum(1 for r in valid_results if r.get("ex_match") is True)
    pass_at_1_count = sum(1 for r in valid_results if r.get("ex_match") is True and r.get("retry_count", 0) == 0)
    
    avg_table_recall = sum(r.get("table_recall", 0) for r in valid_results) / len(valid_results) if valid_results else 0
    avg_table_precision = sum(r.get("table_precision", 0) for r in valid_results) / len(valid_results) if valid_results else 0
    avg_col_recall = sum(r.get("col_recall", 0) for r in valid_results) / len(valid_results) if valid_results else 0
    avg_col_precision = sum(r.get("col_precision", 0) for r in valid_results) / len(valid_results) if valid_results else 0
    
    print("\n" + "="*70)
    print("📈 REPORT METRICHE SOTA FINALE (BIRD & SPIDER)")
    print("="*70)
    print(f"Totale Domande Valutate: {len(valid_results)}")
    
    print("\n--- 🧠 METRICHE DI RAGIONAMENTO (Agenti 2 & 2.5) ---")
    print(f"Table Recall Media:      {avg_table_recall*100:.1f}%")
    print(f"Table Precision Media:   {avg_table_precision*100:.1f}%")
    print(f"Column Recall Media:     {avg_col_recall*100:.1f}%")
    print(f"Column Precision Media:  {avg_col_precision*100:.1f}%")
    
    print("\n--- 🎯 METRICHE DI ESECUZIONE (Agenti 3 & Critic) ---")
    print(f"Execution Accuracy (Sintassi OK): {ex_syntax_count}/{len(valid_results)} ({ex_syntax_count/len(valid_results)*100:.1f}%)")
    print(f"Pass@1 (EX-Match al 1° colpo):    {pass_at_1_count}/{len(valid_results)} ({pass_at_1_count/len(valid_results)*100:.1f}%)")
    print(f"Pass@3 (EX-Match con Critic):     {ex_match_count}/{len(valid_results)} ({ex_match_count/len(valid_results)*100:.1f}%)")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())