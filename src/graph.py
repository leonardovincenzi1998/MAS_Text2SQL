import json
import re
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from .agent import schema_agent
from .models import SchemaDeps, TableSelectionResult
from .database import DatabaseManager

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    db_path: str
    selected_tables: List[str]
    error: Optional[str]

async def schema_extraction_node(state: AgentState):
    query = state['user_query']
    db_manager = DatabaseManager(state['db_path'])
    deps = SchemaDeps(db_manager=db_manager)
    
    try:
        print(f"\n🚀 AVVIO AGENTE per query: '{query}'")
        
        # Esecuzione
        result = await schema_agent.run(
            query, 
            deps=deps,
            model_settings={'temperature': 0.0} 
        )
        

# ---   BLOCCO DEBUG ESTESO ---
        print("\n📜 STORIA DEL RAGIONAMENTO (Step-by-Step):")
        #result.all_messages() contiene tutta la conversazione:
        #User -> Model (chiama tool) -> Tool (risponde) -> Model (risultato finale)
        for i, msg in enumerate(result.all_messages()):
            #Cerchiamo di stampare in modo leggibile a seconda del tipo di messaggio
            kind = getattr(msg, 'kind', 'msg')
            content = getattr(msg, 'content', str(msg))
            
            #Se è una chiamata a un tool (es. list_tables)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"[{i}] 🤖 IL MODELLO CHIAMA TOOL: {[tc.tool_name for tc in msg.tool_calls]}")
            
            #Se è il risultato di un tool
            elif kind == 'tool_return':
                 print(f"[{i}] 🛠️ RISULTATO TOOL: {content}")
            
            #Messaggi normali
            else:
                #Tronco il contenuto se è lunghissimo
                preview = str(content)[:100] + "..." if len(str(content)) > 100 else content
                print(f"[{i}] 🗣️ {kind.upper()}: {content}")
        print("---------------------------------------------------\n")
        # -----------------------------

        raw_output = result.output
        # --- GESTIONE OUTPUT ---
        
        #CASO 1: PydanticAI ha fatto tutto da solo correttamente
        if isinstance(raw_output, TableSelectionResult):
            return {
                "selected_tables": raw_output.relevant_tables,
                "messages": [f"Ragionamento: {raw_output.reasoning}"] 
            }
        
        #CASO 2: Il modello ha restituito una stringa (JSON sporco o Markdown)
        elif isinstance(raw_output, str):
            print(f"⚠️ Rilevata stringa grezza. Tentativo di pulizia e parsing...")
            
            try:
                #1. Rimuoviamo i backticks del markdown (```json ... ```)
                clean_json = raw_output.replace("```json", "").replace("```", "").strip()
                
                #2. Parsiamo la stringa in un dizionario Python
                data_dict = json.loads(clean_json)
                
                #3. Validiamo il dizionario usando il nostro Modello Pydantic
                validated_obj = TableSelectionResult.model_validate(data_dict)
                
                print("✅ Parsing manuale riuscito!")
                return {
                    "selected_tables": validated_obj.relevant_tables,
                    "messages": [f"Ragionamento (Recuperato): {validated_obj.reasoning}"]
                }
                
            except json.JSONDecodeError:
                return {
                    "selected_tables": [],
                    "error": "Il modello ha restituito testo non strutturato impossibile da leggere come JSON.",
                    "messages": [f"Raw output fallito: {raw_output}"]
                }
            except Exception as e:
                return {
                     "selected_tables": [],
                     "error": f"Il JSON era valido ma non rispettava lo schema: {e}",
                     "messages": [f"Raw output: {raw_output}"]
                }

        else:
            return {"error": f"Tipo imprevisto: {type(raw_output)}"}

    except Exception as e:
        return {"error": f"Errore critico: {str(e)}"}

workflow = StateGraph(AgentState)
workflow.add_node("extract_schema", schema_extraction_node)
workflow.set_entry_point("extract_schema")
workflow.add_edge("extract_schema", END)
app = workflow.compile()