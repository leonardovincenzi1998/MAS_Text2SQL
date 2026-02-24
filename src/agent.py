import json
import warnings
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils import expand_selection_with_graph
from src.models import AgentState, ExtractionResult, TableSelectionResult
from src.tools import search_schema_tool

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

LLM_MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct-AWQ' 
BASE_URL = 'http://localhost:8000/v1'
API_KEY = 'EMPTY'

# llm configuration via langchain adapter
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.1
    )

# prompts maintained exactly as original for stability
ENTITY_EXTRACTOR_SYSTEM_PROMPT = """
### ROLE
You are a highly specialized Natural Language to SQL (NL2SQL) Parser. 
Your expertise lies in mapping Italian natural language queries into structured data components that will be use for select relevant tables and columns for SQLite retrieval for a multi-agent system text2sql.

### OPERATIONAL CONSTRAINTS
- INPUT: Italian natural language.
- OUTPUT: Strictly valid JSON.
- LANGUAGE FOR REASONING: English.
- LANGUAGE FOR ENTITIES/INTENT: Italian (must match the Database schema).
- NO CONVERSATIONAL FILLERS: Do not say "Here is the result" or "Sure".
- THINK STEP BY STEP: Focus on understanding the request of the user querying the database. If the request is complicated or long, break it down into pieces and think about what detailed information the user is looking for.

### EXTRACTION LOGIC & HIERARCHY
1. **Intent Extraction**: 
   - Define the primary goal or goals in Italian. 
   - Use standard verbs: “Selezione”, "Conteggio", "Media", "Ricerca", "Raggruppamento”, “Ordinamento”.

2. **Entity & Attribute Mapping**:
   - Identify possible tables (e.g., “Buildings”) and columns (e.g., “Surface area”) that can answer the user's question in detail.
   - **Crucial**: Keep the original Italian terminology. Do NOT translate "Terreni" to "Lands".
   - Handle multi-word entities as a single string but also search lonely (e.g., "Codice Fiscale", "Destinazione d'uso").

3. **SQL Operation Mapping**:
   - **AGGREGATIONS**: If the users ask for aggregations or other operations map "quanto/quanti" to `COUNT`, "media" to `AVG`, "totale/somma" to `SUM`, "massimo" to `MAX`, "minimo" to `MIN`.
   - **FILTERS (WHERE)**: If the user specifies a condition (e.g., "solo quelli a Roma", "maggiore di 100"), extract the condition and map it to `WHERE`.
   - **GROUPING**: If the user asks for grouped results "per ogni" or "raggruppati per", map to `GROUP BY`.
   - **SORTING**: If the user asks for ordered results or limited results "i primi", "i più cari", "in ordine", map to `ORDER BY` + `LIMIT` if necessary.

### JSON SCHEMA
{{
  "reasoning": "Step-by-step logic in English including why specific SQL operators were chosen.",
  "intent": "Concise summary in Italian.",
  "entities": ["list", "of", "italian", "terms"],
  "operations": ["SQL_KEYWORDS"],
  "filters": ["Specific conditions identified, e.g., 'superficie > 100'"]
}}

#####FEW-SHOT EXAMPLES#####

Input: "Quali sono i beni mobili con etichetta con valore 1 come prima apertura"
Output: {{
  "reasoning": "The user wants to retrieve specific assets ('beni mobili') and their values, filtering specifically for those marked as initial opening ('prima apertura'). We need to select the data and apply an exact match filter for this boolean/numeric condition.",
  "intent": "Ricerca beni mobili con valore di prima apertura",
  "entities": ["beni mobili", "etichetta", "valore", "prima apertura"],
  "operations": ["SELECT", "WHERE"],
  "filters": ["prima apertura = 1"]
}}

Input: "Forniscimi l'elenco dei beni attivi, non eliminati"
Output: {{
  "reasoning": "The user is asking for a list of active assets ('beni attivi') and explicitly requires excluding the deleted ones ('non eliminati'). This translates to a standard SELECT with a boolean filter ensuring the 'deleted' flag is false.",
  "intent": "Elenco beni attivi e non eliminati",
  "entities": ["beni attivi", "non eliminati", "eliminato"],
  "operations": ["SELECT", "WHERE"],
  "filters": ["eliminato = 0"]
}}

Input: "Voglio la descrizione dei beni mobili, della specie e sottospecie, e in quale locale ed edficio si trovano, con il tipo etichetta uguale a 'F' e le informazioni su lotto e tipo di etichetta"
Output: {{
  "reasoning": "The query asks for a detailed description of assets ('beni mobili'), their classification ('specie', 'sottospecie'), and spatial location ('locale', 'edificio'). It explicitly applies an exact text match filter on the label type ('tipo etichetta').",
  "intent": "Dettagli, classificazione e ubicazione beni mobili con etichetta specifica",
  "entities": ["descrizione", "beni mobili", "specie", "sottospecie", "locale", "edificio", "tipo etichetta", "lotto"],
  "operations": ["SELECT", "WHERE"],
  "filters": ["tipo etichetta = 'F'"]
}}

Input: "Valori non di prima apertura per data"
Output: {{
  "reasoning": "The user wants to find financial/inventory values ('valori') that are NOT flagged as initial opening ('non di prima apertura'), correlated with a date ('data'). This requires a negative boolean filter on the opening flag.",
  "intent": "Ricerca valori non di prima apertura filtrati per data",
  "entities": ["valori", "prima apertura", "data"],
  "operations": ["SELECT", "WHERE"],
  "filters": ["prima apertura = 0"]
}}
"""

TABLE_SELECTOR_SYSTEM_PROMPT = """
### ROLE
You are a Senior Data Architect specialized in SQL and Database routing. 
Your expertise lies in analyzing Italian natural language queries and selecting the complete subset of tables from a given database schema to answer the query in detail.

### OPERATIONAL CONSTRAINTS
- INPUT: Italian user query and a Candidate Schema (Tables, Columns, Foreign Keys, Samples).
- OUTPUT: Strictly valid JSON.
- LANGUAGE FOR REASONING: English.
- LANGUAGE FOR TABLE NAMES: Must strictly match the exact names provided in the schema (case-sensitive).
- NO CONVERSATIONAL FILLERS: Do not add greetings or extra text.
- THINK STEP BY STEP: Focus heavily on data relationships and table linkages (JOINs) needed to retrieve the requested info.

### SELECTION LOGIC & CRITERIA
1. **Semantic Matching**: Read all column names, descriptions and samples for each table and column. Do NOT assume a table contains data if you don't see the column. If the user asks for an "address", look for tables with "Via", "Civico", "Comune" (e.g., `Edifici`). 
2. **Foreign Key Chaining (CRITICAL)**: Bridge tables (e.g., `MobiliLocali`, `MobiliSottoSpeci`) are never enough to get textual details. You MUST follow the `[FK->Table.Column]` annotations to reach the final descriptive table (e.g., `Locali`, `Speci`).
3. **The ID Rule**: Columns starting with `Id` (e.g., `IdSottoSpecie`) contain ONLY numerical codes. If the user asks for "details", "name", or "description", you CANNOT stop at the ID column. You MUST include the target table.
4. **Discard Noise**: Ignore tables that were retrieved by the semantic search but are irrelevant to the specific user intent.

### JSON SCHEMA
{{
"reasoning": "EXTREMELY SHORT logic (MAX 2-3 SENTENCES). Do not explain discarded tables. Just state the core join path.",
"central_entity": "The exact name of the main driving table representing the core subject (e.g., 'BeniMobili').",
"relevant_tables": ["List", "of", "exact", "table", "names"]
}}

#####FEW-SHOT EXAMPLES#####

Input:
QUERY: "Dimmi in quali stanze si trovano gli armadi e a che piano sono."
SCHEMA: [Context with BeniMobili, MobiliLocali, Locali, Edifici, SottoSpeci...]
Output: {{
"reasoning": "The core entity is 'BeniMobili' (armadi). The user wants to know the room ('stanze') and the floor ('piano'). Looking at the schema, the floor ('Piano') and room description are in the 'Locali' table. To link 'BeniMobili' to 'Locali', we must traverse the bridge table 'MobiliLocali' using 'IdBeneMobile' and 'IdLocale'. No other tables are needed.",
"central_entity": "BeniMobili",
"relevant_tables": ["BeniMobili", "MobiliLocali", "Locali"]
}}

Input: 
QUERY: "Quante schede patrimoniali attive abbiamo inserito nel 2019?"
SCHEMA: [Context with SchedePatrimoniali, TipiValoreInv, Locali...]
Output: {{
"reasoning": "The user asks for a count of active patrimonial cards ('schede patrimoniali attive') filtered by insertion year (2019). The 'SchedePatrimoniali' table contains both the 'IsSchedaAttiva' flag and the 'DTInserimento' date. No foreign keys need to be resolved for descriptions.",
"central_entity": "SchedePatrimoniali",
"relevant_tables": ["SchedePatrimoniali"]
}}

Input:
QUERY: "Quali sono i codici ARCONET dei piani economici usati per le nostre categorie contabili?"
SCHEMA: [Context with Categorie, PianiEcoStatiPatrimoniali, Locali...]
Output: {{   
"reasoning": "The main topic is accounting categories ('Categorie'). To find the ARCONET codes ('codici ARCONET'), we must look at the 'PianiEcoStatiPatrimoniali' table, because 'Categorie' only has an 'IdPianoEcoStatoPatrimoniale' numerical column. We need both tables to resolve the relation.",
"central_entity": "Categorie",
"relevant_tables": ["Categorie", "PianiEcoStatiPatrimoniali"]
}}   
"""

# node 1: entity, operations, and filters extraction
async def run_entity_extractor(state: AgentState) -> Dict[str, Any]:
    print(f"🕵️‍♀️ (Entity Extractor) Analisi query: '{state['user_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = prompt | structured_llm
    
    try:
        extraction: ExtractionResult = await chain.ainvoke({"input": state['user_query']})
        
        # safely handle empty lists
        ops = extraction.operations if extraction.operations else ["None"]
        filtri = extraction.filters if extraction.filters else ["None"]
        
        print(f"   -> 🧠 Ragionamento: {extraction.reasoning}")
        print(f"   -> 🎯 Intento: {extraction.intent}")
        print(f"   -> 🔑 Entità: {extraction.entities}")
        print(f"   -> ⚙️  Operazioni: {ops}")
        print(f"   -> 🗂️  Filtri: {filtri}")
        
        return {
            "extraction_result": extraction,
            "messages": [f"Entità: {extraction.entities} | Filtri: {filtri} | Intento: {extraction.intent}"]
        }
    except Exception as e:
        return {"error": f"Errore Extractor: {str(e)}"}

# transforms the json schema into an optimized pseudo-markdown format for the llm
def format_schema_for_llm(schema_list: list) -> str:
    formatted_tables = []
    
    for tbl in schema_list:
        name = tbl.get("table_name") or tbl.get("table", "Unknown")
        desc = tbl.get("description", tbl.get("desc", ""))
        cols = tbl.get("columns", [])
        fks = tbl.get("foreign_keys", [])
        cat_vals = tbl.get("categorical_values", "")
        samples = tbl.get("column_samples", {})
        
        # quick fk mapping for annotation
        fk_map = {}
        for fk in fks:
            from_col = fk.get("from_column")
            to_tbl = fk.get("to_table_real") or fk.get("to_table_canonical")
            to_col = fk.get("to_column")
            if from_col and to_tbl:
                target = f"{to_tbl}.{to_col}" if to_col else to_tbl
                fk_map[from_col] = target
                
        # build column strings with fk annotations and data samples
        col_tuples = []
        for col in cols:
            col_str = col
            if col in fk_map:
                col_str += f" [FK->{fk_map[col]}]"
            
            if col in samples and samples[col]:
                # clean up newlines to prevent markdown layout breakage
                safe_samples = [str(s).replace('\n', ' ').replace('\r', '') for s in samples[col]]
                col_str += f" (Esempi: {', '.join(safe_samples)})"
                
            col_tuples.append(col_str)
                
        # format as pseudo-markdown
        tbl_md = f"### Tabella: {name}\n"
        if desc: 
            tbl_md += f"Descrizione: {desc}\n"
        tbl_md += f"Colonne: ( {', '.join(col_tuples)} )\n"
        if cat_vals: 
            tbl_md += f"Valori Notevoli:\n{cat_vals}\n"
            
        formatted_tables.append(tbl_md)
        
    return "\n\n".join(formatted_tables)

# node 2: llm table selection based on semantic search and graph auto-filler
async def run_table_selector(state: AgentState) -> Dict[str, Any]:
    print("🔍 (Table Selector) Ricerca tabelle...")
    
    extraction = state.get("extraction_result")
    
    # vector query preparation
    if extraction and extraction.entities:
        # hybrid query combining original question and entities for keyword boost
        entities_str = " ".join(extraction.entities)
        vector_search_query = f"{state['user_query']} {entities_str}"
    else:
        # fallback to the full query
        vector_search_query = state["user_query"]
        
    print(f"   Testo usato per Chroma: '{vector_search_query}'")
    
    # schema retrieval via chromadb tool
    try:
        # retrieve generous number of tables to provide context
        schema_json = search_schema_tool.invoke({"query": vector_search_query, "k": 10})
    except Exception as e:
        return {"error": f"Errore Chroma: {str(e)}"}
    
    # parsing and markdown generation
    try:
        schema_list = json.loads(schema_json) if schema_json else []
        candidate_tables = [t.get("table_name") or t.get("table") for t in schema_list if t.get("table_name") or t.get("table")]
        print(f"📦 (Table Selector) Candidate tables passate all'Agente 2: {len(candidate_tables)}")
        
        schema_markdown = format_schema_for_llm(schema_list)
        
        # save markdown payload for debugging
        with open("debug_schema_markdown.md", "w", encoding="utf-8") as f:
            f.write(schema_markdown)
        print("💾 Markdown passato all'LLM salvato in 'debug_schema_markdown.md'")
        
    except Exception as _e:
        print(f"⚠️ (Table Selector) Impossibile parsare schema_json: {_e}")
        # emergency fallback to raw json string
        schema_markdown = schema_json  
        schema_list = []
    
    # llm selection
    print("🧠 (Table Selector) Filtering intelligente...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", TABLE_SELECTOR_SYSTEM_PROMPT),
        ("human", "### EXECUTION\nQUERY: {query}\nSCHEMA:\n{schema}n\nOutput:")
    ])
    
    structured_llm = llm.with_structured_output(TableSelectionResult)
    chain = prompt | structured_llm
    
    try:
        result: TableSelectionResult = await chain.ainvoke({
            "schema": schema_markdown,
            "query": state["user_query"]
        })
        
        # 1. base llm selection
        llm_selection = result.relevant_tables
        
        # 2. topological auto-filler to add missing bridge tables
        final_selection = expand_selection_with_graph(
            llm_selection,
            schema_json,
            root_table_real=result.central_entity
        )
        
        # 3. calculate additions for debugging
        added_tables = set(final_selection) - set(llm_selection)
        reasoning_log = result.reasoning
        
        if added_tables:
            msg_autofix = f"\n🤖 [AUTO-FIX] Il sistema ha aggiunto tabelle ponte mancanti: {list(added_tables)}"
            reasoning_log += msg_autofix
            print(msg_autofix)
         
        log_msg = f"✅ Tabelle Selezionate: {final_selection}\n🤔 Ragionamento: {reasoning_log}"
        
        return {
            "selected_tables": final_selection,
            "candidate_tables_schema": schema_json,
            "messages": [log_msg]
        }
        
    except Exception as e:
         return {"error": f"Errore Selector LLM: {str(e)}"}