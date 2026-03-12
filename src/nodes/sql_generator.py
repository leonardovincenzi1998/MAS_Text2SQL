from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models import AgentState
from src.database import DatabaseManager
from src.utils import validate_ast_and_format, prune_ddl_ast, format_table_metadata_as_sql_comment
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
                if colonne_scelte_llm:
                    allowed_cols = set(colonne_scelte_llm)
                else:
                    allowed_cols = colonne_pulite
                
                # cleaning AST
                clean_ddl = prune_ddl_ast(raw_ddl, allowed_cols)

                # sql comments generation with metadata
                meta_comment = format_table_metadata_as_sql_comment(tbl_data, allowed_cols)
                ddl_context += f"-- Schema for {table}:\n{clean_ddl}\n{meta_comment}\n\n"
            else:
                ddl_context += f"-- Schema for {table}:\n{raw_ddl}\n\n"

        #DEBUG
        with open("debug_enriched_ddl.sql", "w", encoding="utf-8") as f:
            f.write(ddl_context)
        print("   💾 (SQL Generator) DDL arricchito salvato in 'debug_enriched_ddl.sql'")
        
    except Exception as e:
        return {"error": f"DDL extraction error: {str(e)}"}    
    # RIMOSSO IL SECONDO EXCEPT DUPLICATO

    # format analytical context from agent 1
    extraction = state.get("extraction_result")
    extracted_info = "None"

    if extraction:
        # --- CORREZIONE FORMATTAZIONE ENTITA' ---
        entities_str = "Nessuna"
        if hasattr(extraction, 'entities') and extraction.entities:
            entities_str = ", ".join([f"[{e.category}: '{e.value}']" for e in extraction.entities])
            
        extracted_info = (
            f"Intent: {extraction.intent}\n"
            f"Entities: {entities_str}\n"
            f"Filters: {extraction.filters}"
        )

    entity_hints = state.get("entity_hints", "Nessun hint disponibile sui valori testuali.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_GENERATOR_SYSTEM_PROMPT),
        ("human", "### CONTEXT\n[ENRICHED DDL SCHEMA]\n{ddl_context}\n\n[EXTRACTED INFO]\n{extracted_info}\n\n[EXACT VALUE HINTS FROM VECTOR DB]\n{entity_hints}\n\n### USER QUESTION\n{query}\n\nOutput:")
    ])
    
    chain = prompt | llm_sql

    try:
        response = await chain.ainvoke({
            "ddl_context": ddl_context,
            "extracted_info": extracted_info,
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
                "execution_status": False,  
                "error_traceback": validation_error,
                "messages": [f"❌ Failed SQL Generation (AST):\n{validation_error}"]
            }
        
        print("   ✅ (AST Validator) Sintassi e Schema Linking confermati.")

        return {
            "generated_sql": final_sql,
            "messages": [f"✅ Generated SQL:\n{final_sql}"]
        }
        
    except Exception as e:
        return {"error": f"SQL Generator Error: {str(e)}"}