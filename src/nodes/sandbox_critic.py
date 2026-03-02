import sqlite3
from typing import Dict, Any
from langchain_core.messages import SystemMessage
from src.models import AgentState, CriticResult
from src.database import DatabaseManager
from src.utils import prune_ddl_ast, format_table_metadata_as_sql_comment
from src.config import llm_reasoning
from src.prompts import QUERY_CRITIC_PROMPT

async def run_execution_sandbox(state: AgentState) -> Dict[str, Any]:
    print("🛠️  (Execution Sandbox) Esecuzione query in ambiente isolato...")

    if state.get("execution_status") is False and state.get("error_traceback"):
        print(f"   ⏭️ (Sandbox) Esecuzione saltata a causa di un errore AST precedente.")
        return {}
    
    query = state.get("generated_sql")
    if not query:
        return {"execution_status": False, "error_traceback": "Nessuna query SQL generata da eseguire."}
    
    db_manager = DatabaseManager(state["db_path"])
    conn = db_manager.get_connection()
    
    try:
        cursor = conn.cursor()
        
        # semantic inspection: enforce a LIMIT 10 if not present to avoid huge payloads
        # and to quickly check if the query returns empty data.
        check_query = query.strip().rstrip(";")
        
        if "LIMIT" not in check_query.upper():
            check_query += "\nLIMIT 10"
            
        cursor.execute(check_query)
        rows = cursor.fetchall()
        
        # converting results into dictionaries
        data_sample = [dict(row) for row in rows]
        
        # data Inspection: Checking for ‘empty results’
        if len(data_sample) == 0:
            msg_errore_logico = (
                "L'esecuzione ha avuto successo sintatticamente, ma il risultato è vuoto (0 righe). "
                "Potrebbe esserci un disallineamento nei filtri (WHERE), discrepanze di maiuscole/minuscole "
                "nei valori testuali, o condizioni di JOIN troppo restrittive."
            )
            print("   ⚠️ (Sandbox) Anomalia semantica: Risultato vuoto rilevato.")
            return {
                "execution_status": False, 
                "error_traceback": msg_errore_logico,
                "data_sample": []
            }
            
        print("   ✅ (Sandbox) Esecuzione sintatticamente e logicamente valida.")
        return {
            "execution_status": True,
            "error_traceback": None,
            "data_sample": data_sample
        }
        
    except sqlite3.Error as e:
        # Execution-Guided Feedback: Capture the actual database stack trace
        traceback_str = f"Errore SQLite ({type(e).__name__}): {str(e)}"
        print(f"   ❌ (Sandbox) Errore di Runtime: {traceback_str}")
        return {
            "execution_status": False,
            "error_traceback": traceback_str
        }
    finally:
        conn.close()    

async def run_query_critic(state: AgentState) -> Dict[str, Any]:
    current_retries = state.get("retry_count", 0)
    print(f"🕵️‍♂️ (Query Critic) Tentativo di correzione #{current_retries + 1}...")
    
    # 1. context recovery
    user_query = state["user_query"]
    selected_tables = state["selected_tables"]
    wrong_sql = state.get("generated_sql", "")
    error_traceback = state.get("error_traceback", "Errore sconosciuto")
    
    # 2. reconstruction of the DDL for Critic
    # critic receives ALL columns saved in ingestion, bypassing the Column Selector
    db_manager = DatabaseManager(state["db_path"])

    full_schema_list = state.get("parsed_schema", [])

    critic_ddl_context = ""
    # 1. DDL reconstruction
    for table in selected_tables:
        raw_ddl = db_manager.get_table_ddl(table)
        tbl_data = next((t for t in full_schema_list if t.get("table_name", t.get("table")) == table), None)
        
        if tbl_data:
            colonne_pulite = set(tbl_data.get("columns", []))
            
            clean_ddl = prune_ddl_ast(raw_ddl, colonne_pulite)
            meta_comment = format_table_metadata_as_sql_comment(tbl_data, colonne_pulite)
            
            critic_ddl_context += f"-- Schema for {table}:\n{clean_ddl}\n{meta_comment}\n\n"
        else:
            critic_ddl_context += f"-- Schema for {table}:\n{raw_ddl}\n\n"
    
    # recupero extraction
    extraction = state.get("extraction_result")
    extracted_info = f"Entità: {getattr(extraction, 'entities', [])} | Filtri: {getattr(extraction, 'filters', [])}" if extraction else "Nessuna estrazione."

    # 3. prompt formatting
    prompt = QUERY_CRITIC_PROMPT.format(
        user_query=user_query,
        selected_tables=", ".join(selected_tables),
        extracted_info=extracted_info,
        schema_ddl=critic_ddl_context,
        wrong_sql=wrong_sql,
        error_traceback=error_traceback
    )
    
    #4. invoking LLM with Structured Output
    messages = [SystemMessage(content=prompt)]
    structured_llm = llm_reasoning.with_structured_output(CriticResult).with_retry(stop_after_attempt=3)
    
    try:
        response: CriticResult = await structured_llm.ainvoke(messages)
        
        print(f"   💡 (Critic Plan): {response.correction_plan}")
        print(f"   🔧 (New SQL): {response.corrected_sql}")
        
        # 4. update the status
        return {
            "generated_sql": response.corrected_sql,
            "retry_count": current_retries + 1,
            # reset the sandbox flags for the next cycle
            "execution_status": None,
            "error_traceback": None,
            "data_sample": None
        }
    except Exception as e:
        print(f"   ❌ (Critic) Errore durante la generazione della correzione: {e}")
        return {
            "retry_count": current_retries + 1,
            "error": f"Errore del Critic Agent: {str(e)}"
        }