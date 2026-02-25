import json
import warnings
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils import expand_selection_with_graph
from src.models import AgentState, ExtractionResult, TableSelectionResult
from src.tools import search_schema_tool
from src.database import DatabaseManager

from src.config import LLM_MODEL_NAME, BASE_URL, API_KEY
from src.prompts import (
    ENTITY_EXTRACTOR_SYSTEM_PROMPT,
    TABLE_SELECTOR_SYSTEM_PROMPT,
    SQL_GENERATOR_SYSTEM_PROMPT
)

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

# llm configuration via langchain adapter
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.1
)

# node 1: entity, operations, and filters extraction
async def run_entity_extractor(state: AgentState) -> Dict[str, Any]:
    print(f"🕵️‍♀️ (Entity Extractor) Analisi query: '{state['user_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    structured_llm = llm.with_structured_output(ExtractionResult)
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
        if desc: 
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
    if extraction and extraction.entities:
        # hybrid query combining original question and entities for keyword boost
        entities_str = " ".join(extraction.entities)
        vector_search_query = f"{state['user_query']} {entities_str}"
    else:
        # fallback to the full query
        vector_search_query = state["user_query"]
        
    print(f"   Testo usato per Chroma: '{vector_search_query}'")
    
    # schema retrieval via chromadb tool
    try:
        # retrieve generous number of tables to provide context
        schema_json = search_schema_tool.invoke({"query": vector_search_query, "k": 10})
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
    
    structured_llm = llm.with_structured_output(TableSelectionResult)
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
        
    # inject raw ddl from database for the selected tables
    # we use the direct sqlite connection to guarantee 100% accuracy
    db_manager = DatabaseManager(state["db_path"])
    ddl_context = ""
    try:
        for table in selected_tables:
            ddl_context += f"-- Schema for {table}:\n{db_manager.get_table_ddl(table)}\n\n"
    except Exception as e:
        return {"error": f"DDL extraction error: {str(e)}"}

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
        # handle both string messages and langchain BaseMessage objects
        last_reasoning = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_GENERATOR_SYSTEM_PROMPT),
        ("human", "### CONTEXT\n[DDL SCHEMA]\n{ddl_context}\n\n[EXTRACTED INFO]\n{extracted_info}\n\n[JOIN LOGIC SUGGESTION]\n{reasoning}\n\n### QUERY\n{query}\n\nOutput:")
    ])
    
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({
            "ddl_context": ddl_context,
            "extracted_info": extracted_info,
            "reasoning": last_reasoning,
            "query": state["user_query"]
        })
        
        # clean output to remove any residual markdown injected by the llm
        raw_sql = response.content.strip()
        clean_sql = raw_sql.replace("```sql", "").replace("```sqlite", "").replace("```", "").strip()
        
        return {
            "generated_sql": clean_sql,
            "messages": [f"✅ SQL Generato:\n{clean_sql}"]
        }
        
    except Exception as e:
        return {"error": f"SQL Generator Error: {str(e)}"}