from typing_extensions import Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


#Modello per l'Entity Extractor
class ExtractionResult(BaseModel):
    """
    Output strutturato dell'agente di estrazione entità.
    Identifica l'intento dell'utente, le entità chiave, le operazioni e i filtri.
    """
    reasoning: str = Field(..., description="Step-by-step logic in English including why specific SQL operators were chosen.")
    intent: str = Field(..., description="Una descrizione concisa dell'obiettivo di recupero informazioni dell'utente.")
    entities: List[str] = Field(default_factory=list, description="Lista di entità tangibili o nominate (es. luoghi, nomi tabelle, soggetti tematici).")
    operations: List[str] = Field(default_factory=list, description="Lista di parole o frasi che descrivono operazioni analitiche o logiche da tradurre in linguaggio SQLite (es. 'MEAN', 'AVG', 'COUNT','MAX').")
    filters: List[str] = Field(default_factory=list, description="Specific conditions identified, e.g., 'superficie > 100'.")

#Modello di output atteso dall'LLM per la selezione delle tabelle
class TableSelectionResult(BaseModel):
    central_entity: str = Field(
        description="La tabella principale (Fact Table) attorno a cui ruota la domanda (es. 'BeniMobili' per beni, 'Ammortamenti' per calcoli)."
    )
    reasoning: str = Field(
        description="Spiegazione logica. Specifica quali tabelle usi per i dati e quali per i filtri/join."
    )
    relevant_tables: List[str] = Field(default_factory=list, description="Lista esatta dei nomi delle tabelle selezionate (inclusi i ponti necessari).")
    
    is_ambiguous: bool = Field(
        default=False,
        description="True se la domanda è troppo vaga per selezionare tabelle con certezza."
    )

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    db_path: str
    selected_tables: List[str]
    extraction_result: Optional[ExtractionResult] = None  #Campo per passare i dati al prossimo agente
    candidate_tables_schema: Optional[str] = None #Qui salviamo il JSON grezzo di Chroma
    error: Optional[str]


@dataclass
class SchemaDeps:
    db_manager: Any