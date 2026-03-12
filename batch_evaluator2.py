import argparse
import asyncio
import os
import sqlglot
import json
import re
import warnings
import sqlite3
import gc
import statistics
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
        sql_clean = sql.replace('\n', ' ')
        tables = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)', sql_clean, re.IGNORECASE)
        return set([t.lower() for t in tables])

def compare_execution_results(db_path: str, golden_sql: str, generated_sql: str) -> bool:
    """ SOTA Metric (EX-Match): Esegue entrambe le query e confronta i set di risultati estratti. """
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
    """ Estrae i nomi di colonna in formato 'tabella.colonna' usando l'AST di sqlglot. """
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

def calc_stats(data: list) -> tuple:
    """Calcola (Media, Deviazione Standard) per una lista di numeri."""
    if not data:
        return 0.0, 0.0
    m = statistics.mean(data)
    s = statistics.stdev(data) if len(data) > 1 else 0.0
    return m, s

def write_to_report(filepath: str, content: str):
    """Scrive in append sul file di testo di report."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content + "\n")

async def main():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL agent in BATCH mode for Evaluation")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
    parser.add_argument("--golden_set", type=str, default="set_domande.txt", help="Path to the Golden Set txt file")
    parser.add_argument("--output_json", type=str, default="metriche_sota.json", help="Path for the output JSON results")
    parser.add_argument("--output_txt", type=str, default="report_statistico.txt", help="Path for the human-readable text report")
    parser.add_argument("--runs", type=int, default=10, help="Number of times to run EACH question")
    args = parser.parse_args()

    # Importa il workflow solo dopo aver settato le variabili d'ambiente
    from src.graph import app 

    print("="*70)
    print("📊 AVVIO VALUTAZIONE BATCH STATISTICA - METRICHE SOTA")
    print(f"Iterazioni per domanda: {args.runs}")
    print("="*70)

    eval_dataset = parse_golden_set(args.golden_set)
    if not eval_dataset:
        return
        
    print(f"Trovate {len(eval_dataset)} domande nel file. Inizio elaborazione...\n")
    
    # Inizializza il file TXT sovrascrivendolo
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" REPORT STATISTICO VALUTAZIONE TEXT-TO-SQL\n")
        f.write(f" Iterazioni per domanda: {args.runs}\n")
        f.write("="*80 + "\n\n")

    results_json = []
    all_valid_runs_flat = [] # Per le metriche globali finali

    for item in eval_dataset:
        q_id = item["id"]
        user_query = item["question"]
        golden_sql = item["golden_sql"]
        
        print(f"\n▶️ [{q_id}/{len(eval_dataset)}] '{user_query}'")
        write_to_report(args.output_txt, "="*80 + f"\nDOMANDA {q_id}: {user_query}\nGOLDEN SQL: {golden_sql}\n" + "-"*80)

        max_domanda_retries = 2
        question_runs_data = []

        # LOOP DELLE 15 ITERAZIONI
        for run_idx in range(args.runs):
            print(f"   🔄 Esecuzione {run_idx + 1}/{args.runs}...")
            
            run_result = {
                "run_id": run_idx + 1,
                "generated_sql": None,
                "execution_status": False,
                "ex_match": False,
                "table_recall": 0.0,
                "table_precision": 0.0,
                "col_recall": 0.0,
                "col_precision": 0.0,
                "retry_count": 0,
                "error": None
            }

            for attempt in range(max_domanda_retries):
                current_state = None
                final_state = None
                try:
                    current_state = {
                            "messages": [HumanMessage(content=user_query)],
                            "user_query": user_query,
                            "db_path": args.db,
                            "selected_tables": [],
                            "error": None,
                            "retry_count": 0
                        }
                    
                    # Timeout globale di 4 minuti per evitare blocchi infiniti
                    final_state = await asyncio.wait_for(app.ainvoke(current_state), timeout=240.0)
                    
                    errore_fatale = final_state.get("error") or final_state.get("error_traceback")
                    esecuzione_ok = final_state.get("execution_status")
                    
                    if errore_fatale and not esecuzione_ok and attempt < max_domanda_retries - 1:
                        print(f"      ⚠️ Errore (Tentativo {attempt + 1}). Ritento...")
                        continue

                    generated_sql = final_state.get("generated_sql")
                    execution_status = final_state.get("execution_status", False)
                    selected_tables = final_state.get("selected_tables", [])
                    retry_count = final_state.get("retry_count", 0)
                    selected_cols_dict = final_state.get("selected_columns") or {}
                    
                    # --- CALCOLO METRICHE SOTA ---
                    ex_match = False
                    if execution_status:
                        ex_match = compare_execution_results(args.db, golden_sql, generated_sql)
                    
                    golden_tables = extract_tables_from_sql(golden_sql)
                    golden_tables_lower = set([t.lower() for t in golden_tables])
                    selected_tables_lower = set([t.lower() for t in selected_tables])
                    
                    tp_tab = len(golden_tables_lower.intersection(selected_tables_lower))
                    table_recall = tp_tab / len(golden_tables) if golden_tables else 0.0
                    table_precision = tp_tab / len(selected_tables_lower) if selected_tables_lower else 0.0
                    
                    selected_cols_fq = set()
                    for t_name, cols in selected_cols_dict.items():
                        for col in cols:
                            selected_cols_fq.add(f"{t_name.lower()}.{col.lower()}")
                    
                    golden_cols_fq = extract_columns_from_sql(golden_sql)
                    tp_col = len(golden_cols_fq.intersection(selected_cols_fq))
                    col_recall = tp_col / len(golden_cols_fq) if golden_cols_fq else 0.0
                    col_precision = tp_col / len(selected_cols_fq) if selected_cols_fq else 0.0

                    # Salva risultati
                    run_result.update({
                        "generated_sql": generated_sql,
                        "execution_status": execution_status,
                        "ex_match": ex_match,
                        "table_recall": table_recall,
                        "table_precision": table_precision,
                        "col_recall": col_recall,
                        "col_precision": col_precision,
                        "retry_count": retry_count,
                        "error": errore_fatale
                    })

                    break # Successo, esci dal ciclo dei retries (attempt)

                except asyncio.TimeoutError:
                    if attempt < max_domanda_retries - 1:
                        continue
                    run_result["error"] = "Timeout"
                    break
                except Exception as e:
                    if attempt < max_domanda_retries - 1:
                        continue
                    run_result["error"] = str(e)
                    break
                finally:
                    # ✅ QUESTA È LA GARANZIA ASSOLUTA DI PULIZIA RAM
                    # Viene eseguito sempre, anche dopo break o continue
                    if current_state is not None:
                        del current_state
                    if final_state is not None:
                        del final_state

            # Aggiungi la run alla lista della domanda
            question_runs_data.append(run_result)
            if run_result["error"] != "Timeout":
                all_valid_runs_flat.append(run_result)

            # Scrivi i dettagli della Run sul File TXT
            status_ico = "🏆" if run_result['ex_match'] else ("✅" if run_result['execution_status'] else "❌")
            run_log = f"  [Run {run_idx + 1}] {status_ico}\n"
            run_log += f"    Gen SQL: {run_result['generated_sql']}\n"
            run_log += f"    Metriche: EX-Match: {run_result['ex_match']} | Syntax: {run_result['execution_status']} | "
            run_log += f"TabRecall: {run_result['table_recall']:.2f} | TabPrec: {run_result['table_precision']:.2f} | ColRecall: {run_result['col_recall']:.2f} | ColPrec: {run_result['col_precision']:.2f}"
            if run_result["error"]:
                run_log += f"\n    Errore: {run_result['error']}"
            write_to_report(args.output_txt, run_log)

            print(run_log)
            # Pulizia e raffreddamento per ogni Run per evitare sovraccarico GPU
            gc.collect() 
            await asyncio.sleep(2.0)

        # ==========================================
        # STATISTICHE DELLA SINGOLA DOMANDA
        # ==========================================
        ex_matches = [1.0 if r["ex_match"] else 0.0 for r in question_runs_data if not r["error"] == "Timeout"]
        exec_status = [1.0 if r["execution_status"] else 0.0 for r in question_runs_data if not r["error"] == "Timeout"]
        tab_recalls = [r["table_recall"] for r in question_runs_data if not r["error"] == "Timeout"]
        tab_precs = [r["table_precision"] for r in question_runs_data if not r["error"] == "Timeout"]
        col_recalls = [r["col_recall"] for r in question_runs_data if not r["error"] == "Timeout"]
        col_precs = [r["col_precision"] for r in question_runs_data if not r["error"] == "Timeout"]

        q_stats = {
            "ex_match": calc_stats(ex_matches),
            "execution_status": calc_stats(exec_status),
            "table_recall": calc_stats(tab_recalls),
            "table_precision": calc_stats(tab_precs),
            "col_recall": calc_stats(col_recalls),
            "col_precision": calc_stats(col_precs)
        }

        # Salva in JSON
        results_json.append({
            "id": q_id,
            "question": user_query,
            "golden_sql": golden_sql,
            "runs": question_runs_data,
            "statistics": q_stats
        })

        # Scrivi statistiche domanda su TXT
        stat_log = f"\n  📊 STATISTICHE DOMANDA {q_id}:\n"
        stat_log += f"    - EX-Match:        Media {q_stats['ex_match'][0]*100:.1f}% (σ = {q_stats['ex_match'][1]*100:.1f}%)\n"
        stat_log += f"    - Syntax Accuracy: Media {q_stats['execution_status'][0]*100:.1f}% (σ = {q_stats['execution_status'][1]*100:.1f}%)\n"
        stat_log += f"    - Table Recall:    Media {q_stats['table_recall'][0]*100:.1f}% (σ = {q_stats['table_recall'][1]*100:.1f}%)\n"
        stat_log += f"    - Table Precision: Media {q_stats['table_precision'][0]*100:.1f}% (σ = {q_stats['table_precision'][1]*100:.1f}%)\n"
        stat_log += f"    - Column Recall:   Media {q_stats['col_recall'][0]*100:.1f}% (σ = {q_stats['col_recall'][1]*100:.1f}%)\n"
        stat_log += f"    - Column Precision:Media {q_stats['col_precision'][0]*100:.1f}% (σ = {q_stats['col_precision'][1]*100:.1f}%)"
        write_to_report(args.output_txt, stat_log + "\n")
        
        print(stat_log)

    # Salvataggio JSON Finale
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)
        
    # ==========================================
    # STATISTICHE GLOBALI (TUTTE LE RUN)
    # ==========================================
    if all_valid_runs_flat:
        gl_ex_matches = [1.0 if r["ex_match"] else 0.0 for r in all_valid_runs_flat]
        gl_exec_status = [1.0 if r["execution_status"] else 0.0 for r in all_valid_runs_flat]
        gl_tab_recalls = [r["table_recall"] for r in all_valid_runs_flat]
        gl_tab_precs = [r["table_precision"] for r in all_valid_runs_flat]
        gl_col_recalls = [r["col_recall"] for r in all_valid_runs_flat]
        gl_col_precs = [r["col_precision"] for r in all_valid_runs_flat]

        global_stats = {
            "ex_match": calc_stats(gl_ex_matches),
            "execution_status": calc_stats(gl_exec_status),
            "table_recall": calc_stats(gl_tab_recalls),
            "table_precision": calc_stats(gl_tab_precs),
            "col_recall": calc_stats(gl_col_recalls),
            "col_precision": calc_stats(gl_col_precs)
        }

        final_report = "\n" + "="*80 + "\n"
        final_report += f"📈 REPORT STATISTICO GLOBALE (Su {len(all_valid_runs_flat)} esecuzioni totali)\n"
        final_report += "="*80 + "\n"
        final_report += "--- 🧠 METRICHE DI RAGIONAMENTO (Agenti 2 & 2.5) ---\n"
        final_report += f"Table Recall:      {global_stats['table_recall'][0]*100:.1f}%  ± {global_stats['table_recall'][1]*100:.1f}%\n"
        final_report += f"Table Precision:   {global_stats['table_precision'][0]*100:.1f}%  ± {global_stats['table_precision'][1]*100:.1f}%\n"
        final_report += f"Column Recall:     {global_stats['col_recall'][0]*100:.1f}%  ± {global_stats['col_recall'][1]*100:.1f}%\n"
        final_report += f"Column Precision:  {global_stats['col_precision'][0]*100:.1f}%  ± {global_stats['col_precision'][1]*100:.1f}%\n"
        
        final_report += "\n--- 🎯 METRICHE DI ESECUZIONE (Agenti 3 & Critic) ---\n"
        final_report += f"Execution Accuracy (Sintassi OK): {global_stats['execution_status'][0]*100:.1f}%  ± {global_stats['execution_status'][1]*100:.1f}%\n"
        final_report += f"EX-Match (Accuratezza Logica):    {global_stats['ex_match'][0]*100:.1f}%  ± {global_stats['ex_match'][1]*100:.1f}%\n"
        final_report += "="*80
        
        print(final_report)
        write_to_report(args.output_txt, final_report)

if __name__ == "__main__":
    asyncio.run(main())