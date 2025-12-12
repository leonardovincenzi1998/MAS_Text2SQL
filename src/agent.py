from pydantic import Field
from pydantic_ai import Agent, RunContext
from .models import SchemaDeps, TableSelectionResult
from .config import get_model

# ---------------------------------------------------------
# DEFINIZIONE TOOL
# ---------------------------------------------------------

async def list_tables_tool(ctx: RunContext[SchemaDeps], search_filter: str = "") -> list[str]:
    """Restituisce i nomi delle tabelle presenti nel database. 
    
    Questo tool scopre i nomi delle tabelle presenti nel database.
    
    Args: (Opzionale)
        search_filter
    : La radice di una parola chiave presa dalla richiesta dell'utente.
            Ad esempio data la parola chiave 'vincoli' usa 'vinc'.
            Lascialo vuoto per vedere TUTTE le tabelle.
    
    Returns:
        Una lista di nomi di tabelle che corrispondono al filtro.
    """
    # Il filtro aiuta a ridurre il context bloat su DB grandi.
    return ctx.deps.db_manager.search_tables(search_filter
 or None)

async def get_schema_tool(ctx: RunContext[SchemaDeps], table_name: str) -> str:
    """Restituisce la struttura completa (DDL) di una tabella specifica.
    
    Questo tool scopre i nomi delle colonne e le relazioni (Foreign Keys) di una tabella.
    
    Args:
        table_name: Il nome esatto della tabella da ispezionare.
        
    Returns:
        Una stringa contenente lo statement CREATE TABLE con colonne e tipi.
        Esempio: "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT...)"
    """
    return ctx.deps.db_manager.get_table_ddl(table_name)

# ---------------------------------------------------------
# SYSTEM PROMPT (Best Practice: Describe Actions, Not Implementations)
# ---------------------------------------------------------

SYSTEM_PROMPT = """
Sei un Data Engineer specializzato in Schema Linking.
Il tuo compito è identificare le tabelle SQL rilevanti per la richiesta dell'utente.

PROTOCOLLO OPERATIVO (Segui in ordine):
1. **Mappa**: Usa SEMPRE per primo e SOLO UNA VOLTA `list_tables_tool` per vedere quali tabelle esistono.
1.1 **Recovery**: Se con il filtro restituisce una lista vuota `[]`, DEVI richiamare `list_tables_tool` SENZA filtro per vedere tutto il database. Non arrenderti mai al primo tentativo vuoto.
2. **Ispezione**: Dopo aver ottenuto risposta dal tool `list_tables_tool` e prima di dare una risposta, chiama sempre `get_schema_tool`, per leggerne le colonne delle tabelle che ritieni utili (ottenute da list_tables_tool) e capire se lo sono effettivamente, specificando al tool sempre la tabella che vuoi esplorare.
3. **Output**: Restituisci SOLO l'oggetto JSON finale.

ESEMPIO DI FORMATO (Devi rispondere SOLO così):
{
  "reasoning": "Ho controllato 'order_items' e 'orders'. Seleziono entrambe perché contengono customer_id e order_id .",
  "relevant_tables": ["orders", "order_items"],
  "is_ambiguous": false
}

REGOLE IMPORTANTI:
- NON scrivere frasi discorsive prima o dopo il JSON.
- NON scrivere codice SQL.
- NON inventare nomi di tabelle e colonne.
- NON usare termini inglesi se l'utente ha scritto in italiano.
"""

# Configurazione Agente
schema_agent = Agent[SchemaDeps, TableSelectionResult](
    model=get_model(),
    system_prompt=SYSTEM_PROMPT,
    retries=3
)

# Registrazione Tool
schema_agent.tool(list_tables_tool)
schema_agent.tool(get_schema_tool)