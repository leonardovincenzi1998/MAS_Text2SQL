import json
import warnings
import re
from typing import Dict, List, Optional, Any
import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import expand_selection_with_graph, validate_ast_and_format, prune_ddl_ast, get_schema_with_formatted_columns
from src.models import AgentState, ExtractionResult, TableSelectionResult, CriticResult, ColumnSelectionResult
from src.tools import search_schema_tool
from src.database import DatabaseManager
from src.config import LLM_MODEL_NAME, BASE_URL, API_KEY
from src.prompts import (
    ENTITY_EXTRACTOR_SYSTEM_PROMPT,
    TABLE_SELECTOR_SYSTEM_PROMPT,
    SQL_GENERATOR_SYSTEM_PROMPT,
    QUERY_CRITIC_PROMPT,
    COLUMN_SELECTOR_SYSTEM_PROMPT
)

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

# llm configuration via langchain adapter
# llm = ChatOpenAI(
#     model=LLM_MODEL_NAME,
#     openai_api_base=BASE_URL,
#     openai_api_key=API_KEY,
#     temperature=0.1
# )

# 1. LLM for Agent 1 and 2 (Entity Extraction and Table Selection)
# Used light penalties to avoid reasoning loops
# and max_tokens as an extreme safety valve.
llm_reasoning = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.1,
    #max_tokens=2000,
    presence_penalty=0.2,
    frequency_penalty=0.2
)

# # 2. LLM for Agent 3 (SQL Generation)
# # No penalty SQL need to repeat keywords (JOIN, ON, column names).
llm_sql = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.0
    #max_tokens=1000
)

# node 1: entity, operations, and filters extraction
async def run_entity_extractor(state: AgentState) -> Dict[str, Any]:
    print(f"🕵️‍♀️ (Entity Extractor) Analisi query: '{state['user_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    structured_llm = llm_reasoning.with_structured_output(ExtractionResult).with_retry(stop_after_attempt=3)
    chain = prompt | structured_llm
    
    try:
        extraction: ExtractionResult = await chain.ainvoke({"input": state['user_query']})
        
        # safely handle empty lists
        ops = extraction.operations if extraction.operations else ["None"]
        filtri = extraction.filters if extraction.filters else ["None"]
        
        print(f"   -> 🧠 Ragionamento: {extraction.reasoning}")
        print(f"   -> 🎯 Intento: {extraction.intent}")
        print(f"   -> 🔑 Entità: {extraction.entities}")
        print(f"   -> ⚙️  Operazioni: {ops}")
        print(f"   -> 🗂️  Filtri: {filtri}")
        
        return {
            "extraction_result": extraction,
            "messages": [f"Entità: {extraction.entities} | Filtri: {filtri} | Intento: {extraction.intent}"]
        }
    except Exception as e:
        return {"error": f"Errore Extractor: {str(e)}"}

# transforms the json schema into an optimized pseudo-markdown format for the llm
def format_schema_for_llm(schema_list: list, selected_columns: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Genera il Markdown leggendo le stringhe pre-calcolate, applicando solo il filtro delle colonne.
    """
    formatted_tables = []
    
    for tbl in schema_list:
        name = tbl.get("table_name") or tbl.get("table", "Unknown")
        desc = tbl.get("description", tbl.get("desc", ""))
        cat_vals = tbl.get("categorical_values", "")
        
        # retrieve the pre-calculated cache and raw columns
        formatted_cols_dict = tbl.get("formatted_columns_dict", {})
        raw_columns = tbl.get("columns", [])
        
        # FUNNEL STEP: Column filter
        cols_to_keep = []
        if selected_columns and name in selected_columns:
            sc_upper = [c.upper() for c in selected_columns[name]]
            cols_to_keep = [c for c in raw_columns if c.upper() in sc_upper or c.upper().startswith("ID")]
        else:
            cols_to_keep = raw_columns

        # directly retrieve the string enriched by the dictionary
        col_tuples = [formatted_cols_dict.get(col, col) for col in cols_to_keep]
            
        # pseudo-markdown construction
        tbl_md = f"### Tabella: {name}\n"
        
        if desc and desc.strip() and desc != "Unknown": 
            tbl_md += f"Descrizione: {desc}\n"
            
        tbl_md += f"Colonne: ( {', '.join(col_tuples)} )\n"
        if cat_vals: 
            tbl_md += f"Valori Notevoli:\n{cat_vals}\n"
            
        formatted_tables.append(tbl_md)
        
    return "\n\n".join(formatted_tables)

# node 2: llm table selection based on semantic search and graph auto-filler
async def run_table_selector(state: AgentState) -> Dict[str, Any]:
    print("🔍 (Table Selector) Ricerca tabelle...")
    
    extraction = state.get("extraction_result")
    
    # vector query preparation
    if extraction and hasattr(extraction, 'search_keywords') and extraction.search_keywords:
        # Use keywords optimised by LLM (which now include singular and plural forms)
        keywords_str = " ".join(extraction.search_keywords)
        # Keep the user query for semantic context, but give keywords enormous weight.
        vector_search_query = f"{state['user_query']} {keywords_str} {keywords_str}"
    elif extraction and extraction.entities:
        entities_str = " ".join(extraction.entities)
        vector_search_query = f"{state['user_query']} {entities_str}"
    else:
        vector_search_query = state["user_query"]
        
    print(f"   Testo usato per Chroma: '{vector_search_query}'")
    
    # schema retrieval via chromadb tool
    try:
        # retrieve generous number of tables to provide context
        schema_json = search_schema_tool.invoke({"query": vector_search_query, "k": 6}) #mettere k = 7 con Llama 70B per evitare di sforare i token 
    except Exception as e:
        return {"error": f"Errore Chroma: {str(e)}"}
    
    # parsing and markdown generation
    try:
        schema_list = json.loads(schema_json) if schema_json else []
        schema_list = get_schema_with_formatted_columns(schema_list)
        candidate_tables = [t.get("table_name") or t.get("table") for t in schema_list if t.get("table_name") or t.get("table")]
        print(f"📦 (Table Selector) Candidate tables passate all'Agente 2: {len(candidate_tables)}")
        
        schema_markdown = format_schema_for_llm(schema_list)
        
        # save markdown payload for debugging
        with open("debug_schema_markdown.md", "w", encoding="utf-8") as f:
            f.write(schema_markdown)
        print("💾 Markdown passato all'LLM salvato in 'debug_schema_markdown.md'")
        
    except Exception as _e:
        print(f"⚠️ (Table Selector) Impossibile parsare schema_json: {_e}")
        # emergency fallback to raw json string
        schema_markdown = schema_json  
        schema_list = []
    
    # llm selection
    print("🧠 (Table Selector) Filtering intelligente...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", TABLE_SELECTOR_SYSTEM_PROMPT),
        ("human", "### EXECUTION\nQUERY: {query}\nSCHEMA:\n{schema}\n\nOutput:")
    ])
    
    structured_llm = llm_reasoning.with_structured_output(TableSelectionResult).with_retry(stop_after_attempt=3)
    chain = prompt | structured_llm
    
    try:
        result: TableSelectionResult = await chain.ainvoke({
            "schema": schema_markdown,
            "query": state["user_query"]
        })
        
        # 1. base llm selection
        llm_selection = result.relevant_tables
        
        # 2. topological auto-filler to add missing bridge tables
        final_selection = expand_selection_with_graph(
            llm_selection,
            schema_json,
            root_table_real=result.central_entity
        )
        
        # 3. calculate additions for debugging
        added_tables = set(final_selection) - set(llm_selection)
        reasoning_log = result.reasoning
        
        if added_tables:
            msg_autofix = f"\n🤖 [AUTO-FIX] Il sistema ha aggiunto tabelle ponte mancanti: {list(added_tables)}"
            reasoning_log += msg_autofix
            print(msg_autofix)
         
        log_msg = f"✅ Tabelle Selezionate: {final_selection}\n🤔 Ragionamento: {reasoning_log}"
        
        return {
            "selected_tables": final_selection,
            "parsed_schema": schema_list, 
            "messages": [log_msg]
        }
        
    except Exception as e:
        return {"error": f"Errore Selector LLM: {str(e)}"}

# node 2.5: targeted column selection (schema linking) to prune DDL
async def run_column_selector(state: AgentState) -> Dict[str, Any]:
    print("🎯 (Column Selector) Selezione mirata delle colonne (Schema Linking)...")
    
    selected_tables = state.get("selected_tables", [])
    if not selected_tables:
        return {"error": "Nessuna tabella passata al Column Selector."}

    
    # restrict the JSON to only the selected tables
    full_schema_list = state.get("parsed_schema", [])

    if not full_schema_list:
        return {"error": "Schema parsato non trovato nello stato."}

    selected_schema_list = [
        tbl for tbl in full_schema_list 
        if tbl.get("table_name", tbl.get("table")) in selected_tables
    ]
    markdown_context = format_schema_for_llm(selected_schema_list)


    prompt = ChatPromptTemplate.from_messages([
        ("system", COLUMN_SELECTOR_SYSTEM_PROMPT),
        ("human", "DOMANDA UTENTE: {query}\n\nSCHEMA TABELLE SELEZIONATE:\n{schema}\n\nOutput:")
    ])
    
    structured_llm = llm_reasoning.with_structured_output(ColumnSelectionResult).with_retry(stop_after_attempt=3)
    chain = prompt | structured_llm
    
    try:
        result: ColumnSelectionResult = await chain.ainvoke({
            "query": state["user_query"],
            "schema": markdown_context
        })
        
        print(f"   -> 🧠 Ragionamento: {result.reasoning}")
        print(f"   -> 📎 Colonne Scelte: {result.table_columns}")
        
        return {
            "selected_columns": result.table_columns,
            "messages": [f"✅ Colonne Selezionate:\n{result.table_columns}"]
        }
    except Exception as e:
        return {"error": f"Errore Column Selector LLM: {str(e)}"}
    
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
    if extraction:
        extracted_info = (
            f"Intent: {extraction.intent}\n"
            f"Entities: {extraction.entities}\n"
            f"Operations: {extraction.operations}\n"
            f"Filters: {extraction.filters}"
        )

    # retrieve graph/table selector reasoning from agent 2 to guide joins
    messages = state.get("messages", [])
    last_reasoning = "None"
    if messages:
        last_reasoning = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

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
            "reasoning": last_reasoning,
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
        check_query = query
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

    try:        
        # 1. markdown reconstruction for Critic (only selected tables, but all columns to give it maximum context to understand the error)
        selected_schema_list = [
            tbl for tbl in full_schema_list 
            if tbl.get("table_name", tbl.get("table")) in selected_tables
        ]
        critic_markdown_context = format_schema_for_llm(selected_schema_list) 
    except Exception:
        full_schema_list = []
        critic_markdown_context = "Nessun profilo dati disponibile"

    critic_ddl_context = ""
    # 2. DDL reconstruction
    for table in selected_tables:
        raw_ddl = db_manager.get_table_ddl(table)
        tbl_data = next((t for t in full_schema_list if t.get("table_name", t.get("table")) == table), None)
        
        if tbl_data:
            colonne_pulite = set(tbl_data.get("columns", []))
            
            clean_ddl = prune_ddl_ast(raw_ddl, colonne_pulite)
            critic_ddl_context += f"-- Schema for {table}:\n{clean_ddl}\n\n"
        else:
            critic_ddl_context += f"-- Schema for {table}:\n{raw_ddl}\n\n"
    
    # 3. prompt formatting
    prompt = QUERY_CRITIC_PROMPT.format(
        user_query=user_query,
        selected_tables=", ".join(selected_tables),
        schema_ddl=critic_ddl_context,
        markdown_context=critic_markdown_context,
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