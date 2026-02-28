from typing import Dict, List, Optional, Any, TypedDict
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass
from langgraph.graph.message import add_messages

# Structured output for the entity extraction agent
class ExtractionResult(BaseModel):
    reasoning: str = Field(..., description="Step-by-step logic in English including why specific SQL operators were chosen.")
    intent: str = Field(..., description="Concise description of the user's information retrieval goal.")
    entities: List[str] = Field(default_factory=list, description="List of tangible or named entities (e.g., locations, table names).")
    operations: List[str] = Field(default_factory=list, description="List of analytical operations to translate into SQLite (e.g., 'MEAN', 'COUNT').")
    filters: List[str] = Field(default_factory=list, description="Specific conditions identified, e.g., 'surface > 100'.")

class SearchSchemaInput(BaseModel):
    query: str = Field(description="Entity or keywords to search for.")
    k: int = Field(default=10, description="Number of semantic anchor tables to retrieve.")
    
# Structured output for the table selection agent
class TableSelectionResult(BaseModel):
    central_entity: str = Field(description="The main table that the query revolves around (e.g., 'BeniMobili').")
    reasoning: str = Field(description="EXTREMELY SHORT logical explanation (max 3 sentences) specifying which tables are used.")
    relevant_tables: List[str] = Field(default_factory=list, description="Exact list of selected table names including bridge tables.")
    is_ambiguous: bool = Field(default=False, description="True if the query is too vague to select tables with certainty.")

# Represents the state of the LangGraph multi-agent workflow
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    db_path: str
    selected_tables: List[str]
    extraction_result: Optional[ExtractionResult] = None
    candidate_tables_schema: Optional[str] = None
    generated_sql: Optional[str] = None
    error: Optional[str] = None

    execution_status: Optional[bool] = None     # Track success/failure of execution in Sandbox
    error_traceback: Optional[str] = None       # Caught SQL exception or ‘empty result’ flag
    retry_count: int                            # Counter to prevent infinite loops
    data_sample: Optional[List[Dict[str, Any]]] = None # Extracted data sample (if successful)

class CriticResult(BaseModel):
    correction_plan: str = Field(description="Step-by-step reasoning that identifies the error category (from the taxonomy) and briefly explains how to correct it.")
    corrected_sql: str = Field(description="The new SQL query is correct and ready to be executed, without markdown or comments.")
# Dependencies injected into the schema validation processes
@dataclass
class SchemaDeps:
    db_manager: Any