import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models import AgentState, TableSelectionResult
from src.tools import search_schema_tool
from src.utils import expand_selection_with_graph, get_schema_with_formatted_columns, format_schema_for_llm
from src.config import llm_reasoning
from src.prompts import TABLE_SELECTOR_SYSTEM_PROMPT

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
        entities_str = " ".join([e.value for e in extraction.entities])
        vector_search_query = f"{state['user_query']} {entities_str}"
    else:
        vector_search_query = state["user_query"]
        
    print(f"  Testo usato per Chroma: '{vector_search_query}'")
    
    # schema retrieval via chromadb tool
    try:
        # retrieve generous number of tables to provide context
        schema_json = search_schema_tool.invoke({"query": vector_search_query, "k": 10}) 
    except Exception as e:
        return {"error": f"Chroma error: {str(e)}"}
    
    # parsing and markdown generation
    try:
        schema_list = json.loads(schema_json) if schema_json else []
        schema_list = get_schema_with_formatted_columns(schema_list)
        candidate_tables = [t.get("table_name") or t.get("table") for t in schema_list if t.get("table_name") or t.get("table")]
        print(f"📦 (Table Selector) Candidate tables passate all'Agente 2: {len(candidate_tables)}")
        
        schema_markdown = format_schema_for_llm(schema_list, include_samples=False)
        
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

    ext_context = "No extraction provided."
    if extraction:

        entities_str = "None"
        if hasattr(extraction, 'entities') and extraction.entities:
            entities_str = ", ".join([f"[{e.category}: '{e.value}']" for e in extraction.entities])

        filters_str = "Nessun filtro"
        if hasattr(extraction, 'filters') and extraction.filters:
            filters_str = " | ".join(extraction.filters)

        ext_context = (
            f"- Entities: {entities_str}\n"
            f"- Filters: {filters_str}\n"
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", TABLE_SELECTOR_SYSTEM_PROMPT),
        ("human", "QUERY: {query}\n\n[HINTS FROM AGENT 1]:\n{ext_context}\n\nSCHEMA:\n{schema}")
    ])
    
    structured_llm = llm_reasoning.with_structured_output(TableSelectionResult).with_retry(stop_after_attempt=3)
    chain = prompt | structured_llm
    
    try:
        result: TableSelectionResult = await chain.ainvoke({
            "schema": schema_markdown,
            "query": state["user_query"],
            "ext_context": ext_context
        })

        print(f"   -> 📎 Selected Tables (LLM): {result.relevant_tables}")
        
        # 1. base llm selection
        llm_selection = result.relevant_tables
        
        # 2. topological auto-filler to add missing bridge tables
        final_selection = expand_selection_with_graph(
            llm_selection,
            schema_json,
            root_table_real=result.central_entity
        )
        
        log_msg=""

        # 3. calculate additions for debugging
        added_tables = set(final_selection) - set(llm_selection)
        if added_tables:
            msg_autofix = f"\n🤖 [AUTO-FIX] The system has added missing bridge tables: {list(added_tables)}"
            log_msg += msg_autofix
            print(msg_autofix)
        
        log_msg = f"✅ Selected Tables: {final_selection}\n"
        
        return {
            "selected_tables": final_selection,
            "parsed_schema": schema_list, 
            "messages": [log_msg]
        }
        
    except Exception as e:
        print(f"   ❌ (Table Selector) Errore CRITICO: {str(e)}")
        return {"error": f"Table Selector LLM Error: {str(e)}", "selected_tables": []}