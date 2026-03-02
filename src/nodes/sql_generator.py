from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models import AgentState
from src.database import DatabaseManager
from src.utils import validate_ast_and_format, prune_ddl_ast, format_schema_for_llm
from src.config import llm_sql
from src.prompts import SQL_GENERATOR_SYSTEM_PROMPT

# node 3: sql generation based on injected ddl, entities extraction and table selection
async def run_sql_generator(state: AgentState) -> Dict[str, Any]:
    print("✍️  (SQL Generator) Iniezione DDL e generazione query...")
    
    # retrieve selected tables from previous agent
    selected_tables = state.get("selected_tables", [])

    selected_columns = state.get("selected_columns", {})

    if not selected_tables:
        return {"error": "No tables selected by Agent 2."}
        
    db_manager = DatabaseManager(state["db_path"])
    ddl_context = ""
    
    full_schema_list = state.get("parsed_schema", [])

    #1. clean DDL injection (Smart column pruning)
    try:
        for table in selected_tables:
            raw_ddl = db_manager.get_table_ddl(table)
            tbl_data = next((t for t in full_schema_list if t.get("table_name", t.get("table")) == table), None)
            
            if tbl_data:
                colonne_pulite = set(tbl_data.get("columns", []))
                colonne_scelte_llm = state.get("selected_columns", {}).get(table, [])
                
                # choose the authorised columns:
                # if Agent 2.5 has chosen the columns, use those
                # otherwise, use all the clean ones found during Ingestion as a fallback
                if colonne_scelte_llm:
                    allowed_cols = set(colonne_scelte_llm)
                else:
                    allowed_cols = colonne_pulite
                
                # cleaning AST
                clean_ddl = prune_ddl_ast(raw_ddl, allowed_cols)
                ddl_context += f"-- Schema for {table}:\n{clean_ddl}\n\n"
            else:
                # security fallback: if it cannot find the metadata, pass the entire DDL
                ddl_context += f"-- Schema for {table}:\n{raw_ddl}\n\n"
    except Exception as e:
        return {"error": f"DDL extraction error: {str(e)}"}

    # 2. injection of Profiled Markdown (Only for selected tables)
    try:
        # filter to keep only the tables confirmed by Agent 2
        selected_schema_list = [
            tbl for tbl in full_schema_list 
            if tbl.get("table_name", tbl.get("table")) in selected_tables
        ]
        markdown_context = format_schema_for_llm(selected_schema_list, state.get("selected_columns"))
    except Exception as e:
        print(f"⚠️ Impossibile generare il markdown per l'Agente 3: {e}")
        markdown_context = "Nessun profilo dati disponibile."

    # format analytical context from agent 1
    extraction = state.get("extraction_result")
    extracted_info = "None"
    entity_hints = "Nessun hint disponibile sui valori testuali."

    if extraction:
        extracted_info = (
            f"Intent: {extraction.intent}\n"
            f"Entities: {extraction.entities}\n"
            f"Operations: {extraction.operations}\n"
            f"Filters: {extraction.filters}"
        )

        if extraction.entities:
            print("   🔍 (Value Linking) Risoluzione semantica delle entità a testo libero...")
            from src.tools import resolve_entities_in_db
            entity_hints = resolve_entities_in_db(extraction.entities)

    # retrieve graph/table selector reasoning from agent 2 to guide joins
    messages = state.get("messages", [])
    agent2_reasoning = "Nessun ragionamento Tabelle."
    agent25_reasoning = "Nessun ragionamento Colonne."

    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if "Tabelle Selezionate:" in content:
            agent2_reasoning = content
        elif "Colonne Selezionate:" in content:
            agent25_reasoning = content
            
    combined_reasoning = f"--- RAGIONAMENTO JOIN (Agente 2) ---\n{agent2_reasoning}\n\n--- RAGIONAMENTO COLONNE E FILTRI (Agente 2.5) ---\n{agent25_reasoning}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_GENERATOR_SYSTEM_PROMPT),
        ("human", "### CONTESTO\n[DDL SCHEMA (SINTASSI)]\n{ddl_context}\n\n[PROFILO DATI E VALORI CATEGORICI (MARKDOWN)]\n{markdown_context}\n\n[INFO ESTRATTE]\n{extracted_info}\n\n[SUGGERIMENTO JOIN LOGIC]\n{reasoning}\n\n### DOMANDA UTENTE\n{query}\n\nOutput:")
    ])
    
    chain = prompt | llm_sql

    try:
        response = await chain.ainvoke({
            "ddl_context": ddl_context,
            "markdown_context": markdown_context,
            "extracted_info": extracted_info,
            "reasoning": combined_reasoning,
            "entity_hints": entity_hints,
            "query": state["user_query"]
        })
        
        # clean output to remove any residual markdown injected by the llm
        raw_sql = response.content.strip()
        clean_sql = raw_sql.replace("```sql", "").replace("```sqlite", "").replace("```", "").strip()
        
# execution of the ast validator
        print("   🔍 (AST Validator) Controllo conformità Tabelle e Colonne...")
        final_sql, validation_error = validate_ast_and_format(clean_sql, selected_tables, selected_columns)
        
        if validation_error:
            print("   ⚠️ (AST Validator) Errore rilevato. Invio al Critic Agent.")
            return {
                "generated_sql": final_sql,
                "execution_status": False,  # bypass Sandbox and trigger Critic
                "error_traceback": validation_error,
                "messages": [f"❌ Generazione Fallita (AST):\n{validation_error}"]
            }
        
        print("   ✅ (AST Validator) Sintassi e Schema Linking confermati.")

        return {
            "generated_sql": final_sql,
            "messages": [f"✅ SQL Generato:\n{final_sql}"]
        }
        
    except Exception as e:
        return {"error": f"SQL Generator Error: {str(e)}"}