import json
import warnings
import re
from typing import Dict, Any
import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import expand_selection_with_graph
from src.models import AgentState, ExtractionResult, TableSelectionResult, CriticResult
from src.tools import search_schema_tool
from src.database import DatabaseManager
from src.config import LLM_MODEL_NAME, BASE_URL, API_KEY
from src.prompts import (
    ENTITY_EXTRACTOR_SYSTEM_PROMPT,
    TABLE_SELECTOR_SYSTEM_PROMPT,
    SQL_GENERATOR_SYSTEM_PROMPT,
    QUERY_CRITIC_PROMPT
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
def format_schema_for_llm(schema_list: list) -> str:
    formatted_tables = []
    
    for tbl in schema_list:
        name = tbl.get("table_name") or tbl.get("table", "Unknown")
        desc = tbl.get("description", tbl.get("desc", ""))
        cols = tbl.get("columns", [])
        fks = tbl.get("foreign_keys", [])
        cat_vals = tbl.get("categorical_values", "")
        samples = tbl.get("column_samples", {})
        
        # quick fk mapping for annotation
        fk_map = {}
        for fk in fks:
            from_col = fk.get("from_column")
            to_tbl = fk.get("to_table_real") or fk.get("to_table_canonical")
            to_col = fk.get("to_column")
            if from_col and to_tbl:
                target = f"{to_tbl}.{to_col}" if to_col else to_tbl
                fk_map[from_col] = target
                
        # build column strings with fk annotations and data samples
        col_tuples = []
        for col in cols:
            col_str = col
            if col in fk_map:
                col_str += f" [FK->{fk_map[col]}]"
            
            if col in samples and samples[col]:
                # clean up newlines to prevent markdown layout breakage
                safe_samples = [str(s).replace('\n', ' ').replace('\r', '') for s in samples[col]]
                col_str += f" (Esempi: {', '.join(safe_samples)})"
                
            col_tuples.append(col_str)
                
        # format as pseudo-markdown
        tbl_md = f"### Tabella: {name}\n"
        
        # Print the description if it exists and is valid
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
            "candidate_tables_schema": schema_json,
            "messages": [log_msg]
        }
        
    except Exception as e:
        return {"error": f"Errore Selector LLM: {str(e)}"}

# node 3: sql generation based on injected ddl, entities extraction and table selection
async def run_sql_generator(state: AgentState) -> Dict[str, Any]:
    print("✍️  (SQL Generator) Iniezione DDL e generazione query...")
    
    # retrieve selected tables from previous agent
    selected_tables = state.get("selected_tables", [])
    if not selected_tables:
        return {"error": "No tables selected by Agent 2."}
        
    db_manager = DatabaseManager(state["db_path"])
    ddl_context = ""
    candidate_schema_str = state.get("candidate_tables_schema", "[]")
    
    # Parse securely the metadata saved in ChromaDB (the ‘Slim Schema’ generated in Ingestion)
    try:
        full_schema_list = json.loads(candidate_schema_str)
    except Exception as e:
        print(f"⚠️ Fallito il parsing del candidate_schema_str: {e}")
        full_schema_list = []

    #1. Clean DDL injection (Smart column pruning)
    try:
        for table in selected_tables:
            
            raw_ddl = db_manager.get_table_ddl(table)
            
            # Search for the table in Chroma's metadata to get clean columns
            tbl_data = next((t for t in full_schema_list if t.get("table_name", t.get("table")) == table), None)
            
            if tbl_data:
                colonne_pulite = set(tbl_data.get("columns", []))
                ddl_lines = raw_ddl.split('\n')
                pruned_ddl_lines = []
                
                for line in ddl_lines:
                    line_upper = line.upper()
                    
                    # Always keep the header, brackets, constraints and keys
                    if any(keyword in line_upper for keyword in ["CREATE TABLE", ");", "CONSTRAINT", "PRIMARY KEY", "FOREIGN KEY"]):
                        pruned_ddl_lines.append(line)
                        continue
                        
                    # For column definitions, check whether the column name is in the “clean” list
                    words = line.strip().split()
                    if words:
                        col_name = words[0].replace('"', '').replace('[', '').replace(']', '')
                        if col_name in colonne_pulite:
                            pruned_ddl_lines.append(line)
                
                # Rebuild the thinned-out DDL
                clean_ddl = "\n".join(pruned_ddl_lines)
                ddl_context += f"-- Schema for {table}:\n{clean_ddl}\n\n"
            else:
                # Security fallback: if it cannot find the metadata, pass the entire DDL
                ddl_context += f"-- Schema for {table}:\n{raw_ddl}\n\n"
    except Exception as e:
        return {"error": f"DDL extraction error: {str(e)}"}

    # 2. Injection of Profiled Markdown (Only for selected tables)
    try:
        # Filter to keep only the tables confirmed by Agent 2
        selected_schema_list = [
            tbl for tbl in full_schema_list 
            if tbl.get("table_name", tbl.get("table")) in selected_tables
        ]
        markdown_context = format_schema_for_llm(selected_schema_list)
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
        
        return {
            "generated_sql": clean_sql,
            "pruned_ddl": ddl_context,
            "markdown_context": markdown_context,
            "messages": [f"✅ SQL Generato:\n{clean_sql}"]
        }
        
    except Exception as e:
        return {"error": f"SQL Generator Error: {str(e)}"}
    
async def run_execution_sandbox(state: AgentState) -> Dict[str, Any]:
    print("🛠️  (Execution Sandbox) Esecuzione query in ambiente isolato...")

    query = state.get("generated_sql")
    if not query:
        return {"execution_status": False, "error_traceback": "Nessuna query SQL generata da eseguire."}
    
    db_manager = DatabaseManager(state["db_path"])
    conn = db_manager.get_connection()
    
    try:
        cursor = conn.cursor()
        
        # Semantic Inspection: Enforce a LIMIT 10 if not present to avoid huge payloads
        # and to quickly check if the query returns empty data.
        check_query = query
        if "LIMIT" not in check_query.upper():
            check_query += "\nLIMIT 10"
            
        cursor.execute(check_query)
        rows = cursor.fetchall()
        
        # Converting results into dictionaries
        data_sample = [dict(row) for row in rows]
        
        # Data Inspection: Checking for ‘empty results’
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
    
    # 1. Context recovery
    user_query = state["user_query"]
    selected_tables = state["selected_tables"]
    schema_ddl = state.get("pruned_ddl", "Schema non disponibile")
    markdown_context = state.get("markdown_context", "Nessun profilo dati disponibile")
    wrong_sql = state.get("generated_sql", "")
    error_traceback = state.get("error_traceback", "Errore sconosciuto")
    
    # 2. Prompt formatting
    prompt = QUERY_CRITIC_PROMPT.format(
        user_query=user_query,
        selected_tables=", ".join(selected_tables),
        schema_ddl=schema_ddl,
        markdown_context=markdown_context,
        wrong_sql=wrong_sql,
        error_traceback=error_traceback
    )
    
    #3. Invoking LLM with Structured Output
    messages = [SystemMessage(content=prompt)]
    structured_llm = llm_reasoning.with_structured_output(CriticResult).with_retry(stop_after_attempt=3)
    
    try:
        response: CriticResult = await structured_llm.ainvoke(messages)
        
        print(f"   💡 (Critic Plan): {response.correction_plan}")
        print(f"   🔧 (New SQL): {response.corrected_sql}")
        
        # 4. Update the status
        return {
            "generated_sql": response.corrected_sql,
            "retry_count": current_retries + 1,
            # Reset the sandbox flags for the next cycle
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