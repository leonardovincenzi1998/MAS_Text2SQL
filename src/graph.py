from langgraph.graph import StateGraph, END
from src.models import AgentState

from src.nodes.extractor import run_entity_extractor
from src.nodes.extractor import run_entity_extractor
from src.nodes.table_selector import run_table_selector
from src.nodes.column_selector import run_column_selector
from src.nodes.sql_generator import run_sql_generator
from src.nodes.sandbox_critic import run_execution_sandbox, run_query_critic
from src.nodes.value_linker import run_value_linker
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

def check_error(state: AgentState) -> str:
    """Interrompe il grafo se l'Entity Extractor fallisce criticamente."""
    if state.get("error"):
        print("🛑 (Router) Errore critico rilevato. Interruzione grafo.")
        return "end"
    return "continue"


# conditional routing function
def routing_decision(state: AgentState) -> str:
    """Decide whether to end or invoke the Critic agent based on the results of the Sandbox."""
    status = state.get("execution_status")
    retries = state.get("retry_count", 0)
    MAX_RETRIES = 3 # limit to avoid infinite loops
    
    if status is True:
        # no errors, data found
        return "end"
    
    if retries >= MAX_RETRIES:
        print("🛑 (Router) Numero massimo di tentativi raggiunto. Abortire.")
        return "end"
        
    # error detected (syntactic or semantic), let's send the query to the critic
    return "critic"

# builds and compiles the langgraph workflow for the text-to-sql multi-agent system
def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # add the agents as nodes
    workflow.add_node("entity_extractor", run_entity_extractor)
    workflow.add_node("value_linker", run_value_linker)
    workflow.add_node("table_selector", run_table_selector)
    workflow.add_node("column_selector", run_column_selector)
    workflow.add_node("sql_generator", run_sql_generator)

    workflow.add_node("execution_sandbox", run_execution_sandbox)
    workflow.add_node("query_critic", run_query_critic)

    # define the initial execution flow
    workflow.set_entry_point("entity_extractor")
    workflow.add_conditional_edges(
        "entity_extractor",
        check_error,
        {
            "end": END,
            "continue": "value_linker"
        }
    )
    workflow.add_edge("value_linker", "table_selector")
    workflow.add_edge("table_selector", "column_selector")
    workflow.add_edge("column_selector", "sql_generator")
    workflow.add_edge("sql_generator", "execution_sandbox")

    workflow.add_conditional_edges(
        "execution_sandbox", # starting point
        routing_decision,    # logic function
        {
            "end": END,                  # if True or Max Retries -> End
            "critic": "query_critic"     # otherwise -> Go to 4th Agent
        }
    )
    # end of loop: Critic regenerates the SQL and sends it back to be executed.
    workflow.add_edge("query_critic", "execution_sandbox")
    return workflow.compile()

# compile the app to be imported by interactive_main.py
app = create_workflow()