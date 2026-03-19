import re
import sqlite3
from typing import Dict, Any
from langchain_core.messages import SystemMessage
from src.models import AgentState
from src.database import DatabaseManager
from src.utils import prune_ddl_ast, format_table_metadata_as_sql_comment, validate_ast_and_format
from src.config import llm_reasoning
from src.prompts import QUERY_CRITIC_PROMPT

async def run_execution_sandbox(state: AgentState) -> Dict[str, Any]:
    print("🛠️  (Execution Sandbox) Esecuzione query in ambiente isolato...")

    if state.get("execution_status") is False and state.get("error_traceback"):
        print(f"   ⏭️ (Sandbox) Esecuzione saltata a causa di un errore AST precedente.")
        return {}
    
    query = state.get("generated_sql")
    if not query:
        return {"execution_status": False, "error_traceback": "No SQL query generated to execute."}
    
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
                "The execution succeeded syntactically, but the result is empty (0 rows). "
                "There might be a misalignment in the filters (WHERE), discrepancies in uppercase/lowercase "
                "in the textual values, or JOIN conditions too restrictive."
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
        traceback_str = f"SQLite error ({type(e).__name__}): {str(e)}"
        print(f"   ❌ (Sandbox) Errore di Runtime: {traceback_str}")
        return {
            "execution_status": False,
            "error_traceback": traceback_str
        }
    finally:
        conn.close()    

async def run_query_critic(state: AgentState) -> Dict[str, Any]:

    SQL_REGEX = r"```(?:sql|sqlite)?\s*(.*?)```"

    current_retries = state.get("retry_count", 0)
    print(f"🕵️‍♂️ (Query Critic) Tentativo di correzione #{current_retries + 1}...")
    
    # 1. context recovery
    user_query = state["user_query"]
    selected_tables = state["selected_tables"]
    wrong_sql = state.get("generated_sql", "")
    error_traceback = state.get("error_traceback", "Unknown error")
    
    # 2. reconstruction of the DDL for Critic
    # critic receives ALL columns saved in ingestion, bypassing the Column Selector
    db_manager = DatabaseManager(state["db_path"])

    full_schema_list = state.get("parsed_schema", [])

    critic_ddl_context = ""
    # DDL reconstruction
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
    
    extraction = state.get("extraction_result")
    # load hints directly from state
    entity_hints = state.get("entity_hints", "Nessun hint disponibile.") 
    
    extracted_info = "No extraction available."
    if extraction:
        entities_str = "None"
        if hasattr(extraction, 'entities') and extraction.entities:
            entities_str = ", ".join([f"[{e.category}: '{e.value}']" for e in extraction.entities])
            
        # include the hints in the Critic's prompt
        extracted_info = (
            f"Intent: {getattr(extraction, 'intent', 'Non specificato')}\n"
            f"Entities: {entities_str}\n"
            f"Filters: {getattr(extraction, 'filters', [])}\n"
            f"EXACT VALUE HINTS: {entity_hints}" 
        )
    # 3. prompt formatting
    prompt = QUERY_CRITIC_PROMPT.format(
        user_query=user_query,
        selected_tables=", ".join(selected_tables),
        extracted_info=extracted_info,
        schema_ddl=critic_ddl_context,
        wrong_sql=wrong_sql,
        error_traceback=error_traceback
    )
    
    # 4. invoking LLM with Structured Output
    messages = [SystemMessage(content=prompt)]
    
    try:
        response = await llm_reasoning.ainvoke(messages)
        
        llm_output = response.content if hasattr(response, "content") else str(response)

        print("\n🧠 (Critic Reasoning):\n")
        print(llm_output)

        match = re.search(SQL_REGEX, llm_output, re.DOTALL | re.IGNORECASE)

        if not match:

            return {
                "retry_count": current_retries + 1,
                "error": "SQL block not found in critic output"
            }

        clean_sql = match.group(1).strip()

        print(f"\n   🔧 (New SQL): {clean_sql}")
        print("   🔍 (AST Validator - Critic) Controllo conformità Tabelle e Colonne...")

        # Build a permissive dictionary for the Critic: 
        # contains ALL the real columns of the selected tables, ignoring the Column Selector's output.
        critic_allowed_columns = {}
        for table in selected_tables:
            tbl_data = next((t for t in full_schema_list if t.get("table_name", t.get("table")) == table), None)
            if tbl_data:
                critic_allowed_columns[table] = tbl_data.get("columns", [])

        # validate Critic's query 
        final_sql, validation_error = validate_ast_and_format(
            clean_sql,
            selected_tables,
            selected_columns=critic_allowed_columns # Passiamo lo schema completo
        )

        if validation_error:
            print(f"   ⚠️ (AST Validator - Critic) Errore rilevato: {validation_error}")
            # if AST fails, we set execution_status=False and error_traceback.
            # On the next cycle, the graph will skip the Sandbox and return directly to the Critic!
            return {
            "generated_sql": final_sql,
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
            "error": f"Critic Agent Error: {str(e)}"
        }