import warnings
from typing import Dict, List, Optional, Any, TypedDict
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass
from langgraph.graph.message import add_messages

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")


class ExtractedEntity(BaseModel):
    category: str = Field(description="The conceptual category of the entity (e.g. 'SottoSpecie', 'Locale', 'Anno', 'CondizioneGiuridica', 'Valore').")
    value: str = Field(description="The exact value mentioned by the user (e.g., 'scrivanie in legno', 'Seminterrato', '2022', 'ABC').")

# Structured output for the entity extraction agent
class ExtractionResult(BaseModel):
    intent: str = Field(..., description="Concise description of the user's information retrieval goal.")
    entities: List[ExtractedEntity] = Field(
        default_factory=list, 
        description="List of key entities extracted, categorised by type."
    )
    search_keywords: List[str] = Field(default_factory=list, description="SEO-like keywords optimized for DB search (includes singular/plural forms and synonyms).")
    filters: List[str] = Field(default_factory=list, description="Specific conditions requested by the user IN NATURAL LANGUAGE, e.g., 'Il bene deve essere attivo'. DO NOT use SQL syntax.")

class SearchSchemaInput(BaseModel):
    query: str = Field(description="Entity or keywords to search for.")
    k: int = Field(default=10, description="Number of semantic anchor tables to retrieve.")
    
# Structured output for the table selection agent
class TableSelectionResult(BaseModel):
    central_entity: str = Field(description="The main table that the query revolves around (e.g., 'BeniMobili').")
    relevant_tables: List[str] = Field(default_factory=list, description="Exact list of selected table names including bridge tables.")

# Structured output for the column selection agent (Agent 2.5)
class ColumnSelectionResult(BaseModel):
    #reasoning_steps: List[str] = Field(default_factory=list,description="Extremely short reasoning about the selected columns based on the user's query")
    table_columns: Dict[str, List[str]] = Field(description="Dictionary with exact “table_name” as key and list of exact “column_names” as value.")
    
# Represents the state of the LangGraph multi-agent workflow
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    db_path: str
    selected_tables: List[str]
    selected_columns: Optional[Dict[str, List[str]]] = None
    extraction_result: Optional[ExtractionResult] = None
    parsed_schema: Optional[List[Dict[str, Any]]] = None
    generated_sql: Optional[str] = None
    error: Optional[str] = None

    execution_status: Optional[bool] = None     # Track success/failure of execution in Sandbox
    error_traceback: Optional[str] = None       # Caught SQL exception or ‘empty result’ flag
    retry_count: int                            # Counter to prevent infinite loops
    data_sample: Optional[List[Dict[str, Any]]] = None # Extracted data sample (if successful)
    entity_hints: str
    
# Dependencies injected into the schema validation processes
@dataclass
class SchemaDeps:
    db_manager: Any