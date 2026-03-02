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

# load context from previous agent for better reasoning
    extraction = state.get("extraction_result")
    agent1_context = "Nessun dato estratto dall'Agente 1."
    if extraction:
        agent1_context = (
            f"- Intento: {getattr(extraction, 'intent', 'N/A')}\n"
            f"- Entità: {getattr(extraction, 'entities', [])}\n"
            f"- Operazioni: {getattr(extraction, 'operations', [])}\n"
            f"- Filtri: {getattr(extraction, 'filters', [])}"
        )

    # --- RECUPERO CONTESTO AGENTE 2 ---
    agent2_reasoning = "Ragionamento non disponibile."
    messages = state.get("messages", [])
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if "✅ Tabelle Selezionate:" in content and "🤔 Ragionamento:" in content:
            # Estraiamo solo la parte di ragionamento pulita
            agent2_reasoning = content.split("🤔 Ragionamento:")[-1].strip()
            break

    # --- COSTRUZIONE DEL PROMPT HUMAN ARRICCHITO ---
    human_message_content = f"""DOMANDA UTENTE ORIGINALE: {state['user_query']}
    [CONTEXT AGENT 1 - SEMANTIC EXTRACTION]
    {agent1_context}

    [CONTEXT AGENT 2 - TABLE SELECTION]
    Reasoning behind the choice of tables: {agent2_reasoning}

    [SCHEMA OF SELECTED TABLES]
    {markdown_context}

    Output:"""

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
        
        print(f"   -> 🧠 Ragionamento: {result.reasoning}")
        print(f"   -> 📎 Colonne Scelte: {result.table_columns}")
        
        return {
            "selected_columns": result.table_columns,
            "messages": [f"✅ Colonne Selezionate:\n{result.table_columns}"]
        }
    except Exception as e:
        return {"error": f"Errore Column Selector LLM: {str(e)}"}