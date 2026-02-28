from langgraph.graph import StateGraph, END
from src.agent import run_entity_extractor, run_table_selector, run_sql_generator, run_execution_sandbox, run_query_critic
from src.models import AgentState

# Conditional routing function
def routing_decision(state: AgentState) -> str:
    """Decide se terminare, o invocare l'agente Critic in base ai risultati della Sandbox."""
    status = state.get("execution_status")
    retries = state.get("retry_count", 0)
    MAX_RETRIES = 3 # Limit to avoid infinite loops
    
    if status is True:
        # No errors, data found
        return "end"
    
    if retries >= MAX_RETRIES:
        print("🛑 (Router) Numero massimo di tentativi raggiunto. Abortire.")
        return "end"
        
    # Error detected (syntactic or semantic), let's send the query to the critic
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
        "execution_sandbox", # Starting point
        routing_decision,    # Logic function
        {
            "end": END,                  # If True or Max Retries -> End
            "critic": "query_critic"     # Otherwise -> Go to 4th Agent
        }
    )
    # End of loop: Critic regenerates the SQL and sends it back to be executed.
    workflow.add_edge("query_critic", "execution_sandbox")
    return workflow.compile()

# compile the app to be imported by interactive_main.py
app = create_workflow()