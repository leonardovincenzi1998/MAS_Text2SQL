from typing_extensions import Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


#Modello per l'Entity Extractor
class ExtractionResult(BaseModel):
    """
    Output strutturato dell'agente di estrazione entità.
    Identifica l'intento dell'utente, le entità chiave e le operazioni richieste.
    """
    intent: str = Field(..., description="Una descrizione concisa dell'obiettivo di recupero informazioni dell'utente.")
    entities: List[str] = Field(..., description="Lista di entità tangibili o nominate (es. luoghi, nomi tabelle, soggetti tematici).")
    operations: List[str] = Field(..., description="Lista di parole o frasi che descrivono operazioni analitiche o logiche (es. 'media', 'vicino a', 'conta').")

#Modello di output atteso dall'LLM
class TableSelectionResult(BaseModel):
    reasoning: str = Field(description="Spiegazione logica del perché queste tabelle sono necessarie.")
    relevant_tables: List[str] = Field(description="Lista esatta dei nomi delle tabelle da usare.")
    is_ambiguous: bool = Field(default=False, description="True se servono chiarimenti dall'utente.")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    db_path: str
    selected_tables: List[str]
    extraction_result: Optional[ExtractionResult] = None  #Campo per passare i dati al prossimo agente
    candidate_tables_schema: Optional[str] = None #Qui salviamo il JSON grezzo di Chroma
    error: Optional[str]



#Container per Dependency Injection
@dataclass
class SchemaDeps:
    db_manager: Any