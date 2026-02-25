from langgraph.graph import StateGraph, END
from src.agent import run_entity_extractor, run_table_selector, run_sql_generator
from src.models import AgentState

# builds and compiles the langgraph workflow for the text-to-sql multi-agent system
def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # add the agents as nodes
    workflow.add_node("entity_extractor", run_entity_extractor)
    workflow.add_node("table_selector", run_table_selector)
    workflow.add_node("sql_generator", run_sql_generator)

    # define the execution flow
    workflow.set_entry_point("entity_extractor")
    workflow.add_edge("entity_extractor", "table_selector")
    
    # connect table selector to sql generator
    workflow.add_edge("table_selector", "sql_generator")
    
    # current pipeline finishes after sql generation
    workflow.add_edge("sql_generator", END)

    return workflow.compile()

# compile the app to be imported by interactive_main.py
app = create_workflow()