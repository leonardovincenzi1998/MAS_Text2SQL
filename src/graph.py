from langgraph.graph import StateGraph, END
from src.agent import run_entity_extractor, run_table_selector, run_sql_generator, run_execution_sandbox
from src.models import AgentState

# Funzione dummy per il Critic Agent (la implementeremo nel prossimo step)
async def run_query_critic(state: AgentState):
    # Incrementa il contatore a ogni passaggio nel nodo critic
    current_retries = state.get("retry_count", 0)
    print(f"🕵️‍♂️ (Query Critic) Tentativo di correzione #{current_retries + 1}...")
    # Qui andrà la logica LLM con la Tassonomia degli Errori
    return {"retry_count": current_retries + 1}

# --- FUNZIONE DI ROUTING CONDIZIONALE ---
def routing_decision(state: AgentState) -> str:
    """Decide se terminare, o invocare l'agente Critic in base ai risultati della Sandbox."""
    status = state.get("execution_status")
    retries = state.get("retry_count", 0)
    MAX_RETRIES = 3 # Limite per evitare loop infiniti
    
    if status is True:
        # Nessun errore, dati trovati
        return "end"
    
    if retries >= MAX_RETRIES:
        # Fallimento irreversibile (max tentativi raggiunti)
        print("🛑 (Router) Numero massimo di tentativi raggiunto. Abortire.")
        return "end"
        
    # Errore rilevato (sintattico o semantico), mandiamo la query al critic
    return "critic"

# builds and compiles the langgraph workflow for the text-to-sql multi-agent system
def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # add the agents as nodes
    workflow.add_node("entity_extractor", run_entity_extractor)
    workflow.add_node("table_selector", run_table_selector)
    workflow.add_node("sql_generator", run_sql_generator)

    workflow.add_node("execution_sandbox", run_execution_sandbox)
    workflow.add_node("query_critic", run_query_critic)

    # define the initial execution flow
    workflow.set_entry_point("entity_extractor")
    workflow.add_edge("entity_extractor", "table_selector")
    workflow.add_edge("table_selector", "sql_generator")
    workflow.add_edge("sql_generator", "execution_sandbox")

    workflow.add_conditional_edges(
        "execution_sandbox", # Nodo di partenza
        routing_decision,    # Funzione logica
        {
            "end": END,                  # Se True o Max Retries -> Fine
            "critic": "query_critic"     # Altrimenti -> Vai al 4° Agente
        }
    )
    # 5. Chiusura del ciclo: il Critic rigenera l'SQL e lo rimanda in esecuzione
    workflow.add_edge("query_critic", "execution_sandbox")
    return workflow.compile()

# compile the app to be imported by interactive_main.py
app = create_workflow()