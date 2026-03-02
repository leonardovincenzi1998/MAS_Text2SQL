from typing import Dict, Any
from src.models import AgentState
from src.tools import resolve_entities_in_db

async def run_value_linker(state: AgentState) -> Dict[str, Any]:
    print("   🔍 (Value Linker) Risoluzione semantica delle entità a testo libero...")
    
    extraction = state.get("extraction_result")
    entity_hints = "Nessun hint disponibile sui valori testuali."
    
    # check whether Agent 1 has actually extracted entities
    if extraction and hasattr(extraction, 'entities') and extraction.entities:
        try:
            # Chiama il tool di risoluzione sul database
            entity_hints = resolve_entities_in_db(extraction.entities)
            print("   ✅ (Value Linker) Entity Hints generati con successo.")
        except Exception as e:
            print(f"   ⚠️ (Value Linker) Errore durante la risoluzione: {str(e)}")
            
    # update the global status with the hints found
    return {
        "entity_hints": entity_hints,
        "messages": ["Risoluzione entità completata."]
    }