from typing import Dict, Any
from src.models import AgentState
from src.tools import resolve_entities_in_db

async def run_value_linker(state: AgentState) -> Dict[str, Any]:
    print("   🔍 (Value Linker) Risoluzione semantica delle entità a testo libero...")
    
    extraction = state.get("extraction_result")
    entity_hints = "No exact value hints available."
    
    # check whether Agent 1 has actually extracted entities
    if extraction and hasattr(extraction, 'entities') and extraction.entities:
        try:
            hints_dal_db = resolve_entities_in_db(extraction.entities)
            
            if "No exact" not in hints_dal_db and "No textual" not in hints_dal_db and "Unable" not in hints_dal_db:
                entity_hints = hints_dal_db
                print("   ✅ (Value Linker) Entity Hints generati con successo.")
            else:
                print("   ⚠️ (Value Linker) Nessun match testuale rilevante trovato nel DB.")
                
        except Exception as e:
            print(f"   ⚠️ (Value Linker) Errore durante la risoluzione: {str(e)}")
            
    return {
        "entity_hints": entity_hints,
        "messages": ["Entity resolution completed."]
    }