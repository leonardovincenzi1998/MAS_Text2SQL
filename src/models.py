from typing import List, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass

#Modello di output atteso dall'LLM
class TableSelectionResult(BaseModel):
    reasoning: str = Field(description="Spiegazione logica del perché queste tabelle sono necessarie.")
    relevant_tables: List[str] = Field(description="Lista esatta dei nomi delle tabelle da usare.")
    is_ambiguous: bool = Field(default=False, description="True se servono chiarimenti dall'utente.")

#Container per Dependency Injection
@dataclass
class SchemaDeps:
    db_manager: Any