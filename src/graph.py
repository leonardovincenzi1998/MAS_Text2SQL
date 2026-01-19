import json
import re
from langgraph.graph import StateGraph, END
from .agent import run_entity_extractor, run_table_selector
from .models import AgentState, SchemaDeps, TableSelectionResult
from .database import DatabaseManager

# Costruzione del grafo
workflow = StateGraph(AgentState)

# Aggiungi i nodi
workflow.add_node("entity_extractor", run_entity_extractor)
workflow.add_node("table_selector", run_table_selector)

# Definisci il flusso
workflow.set_entry_point("entity_extractor")
# Dall'estrazione passiamo alla selezione tabelle
workflow.add_edge("entity_extractor", "table_selector")

#Dopo la selezione, finiamo
workflow.add_edge("table_selector", END)

app = workflow.compile()