from langgraph.graph import StateGraph, END
from src.agent import run_entity_extractor, run_table_selector
from src.models import AgentState

# builds and compiles the langgraph workflow for the text-to-sql multi-agent system
def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # add the agents as nodes
    workflow.add_node("entity_extractor", run_entity_extractor)
    workflow.add_node("table_selector", run_table_selector)

    # define the execution flow
    workflow.set_entry_point("entity_extractor")
    workflow.add_edge("entity_extractor", "table_selector")
    
    # current pipeline finishes after table selection
    workflow.add_edge("table_selector", END)

    return workflow.compile()

# compile the app to be imported by interactive_main.py
app = create_workflow()