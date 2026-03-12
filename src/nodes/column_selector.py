from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models import AgentState, ColumnSelectionResult
from src.utils import format_schema_for_llm
from src.config import llm_reasoning
from src.prompts import COLUMN_SELECTOR_SYSTEM_PROMPT

# node 2.5: targeted column selection (schema linking) to prune DDL
async def run_column_selector(state: AgentState) -> Dict[str, Any]:
    print("🎯 (Column Selector) Selezione mirata delle colonne (Schema Linking)...")
    
    selected_tables = state.get("selected_tables", [])
    if not selected_tables:
        return {"error": "No tables passed to Column Selector."}    
    
    # restrict the JSON to only the selected tables
    full_schema_list = state.get("parsed_schema", [])

    if not full_schema_list:
        return {"error": "Parsed schema not found in state."}

    selected_schema_list = [
        tbl for tbl in full_schema_list 
        if tbl.get("table_name", tbl.get("table")) in selected_tables
    ]
    markdown_context = format_schema_for_llm(selected_schema_list)


    entity_hints = state.get("entity_hints", "No exact value hints available.")

    # --- COSTRUZIONE DEL PROMPT HUMAN ARRICCHITO ---
    human_message_content = f"""ORIGINAL USER QUERY: {state['user_query']}

    [HINTS FOR EXACT VALUES FROM VECTOR DB]
    {entity_hints}

    [SCHEMA OF SELECTED TABLES]
    {markdown_context}

    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", COLUMN_SELECTOR_SYSTEM_PROMPT),
        ("human", "{formatted_input}")
    ])
    
    structured_llm = llm_reasoning.with_structured_output(ColumnSelectionResult).with_retry(stop_after_attempt=3)
    chain = prompt | structured_llm
    
    try:
        result: ColumnSelectionResult = await chain.ainvoke({
            "formatted_input": human_message_content
        })
        
        print(f"   -> 📎 Selected Columns: {result.table_columns}")
        
        return {
            "selected_columns": result.table_columns,
            "messages": [f"✅ Selected Columns:\n{result.table_columns}"]
        }
    except Exception as e:
        print(f"   ❌ (Column Selector) Error: {str(e)}")
        return {"error": f"Column Selector LLM Error: {str(e)}"}