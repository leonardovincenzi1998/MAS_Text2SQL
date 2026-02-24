import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from src.graph_utils import expand_selection_with_graph
# Importiamo la configurazione, i modelli e i tools
from src.config import BASE_URL, API_KEY, LLM_MODEL_NAME
from src.models import AgentState, ExtractionResult, TableSelectionResult
from src.tools import search_schema_tool
import warnings
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
# ---------------------------------------------------------
# 1. CONFIGURAZIONE LLM (LangChain Adapter)
# ---------------------------------------------------------
# Usiamo le variabili di config.py per connetterci a Ollama/Cluster
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.1
)

# ---------------------------------------------------------
# 2. PROMPT (La tua logica originale, adattata)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. FUNZIONI DEI NODI (Logic del Grafo)
# ---------------------------------------------------------

async def run_entity_extractor(state: AgentState):
    """
    NODO 1: Estrazione Entità, Operazioni e Filtri.
    """
    print(f"🕵️‍♀️ (Entity Extractor) Analisi query: '{state['user_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = prompt | structured_llm
    
    try:
        extraction = await chain.ainvoke({"input": state['user_query']})
        
        # Gestione sicura delle liste vuote
        ops = extraction.operations if extraction.operations else "Nessuna"
        filtri = extraction.filters if extraction.filters else "Nessuno"
        
        # Log dettagliato con i nuovi campi richiesti dal prompt
        print(f"   -> 🧠 Ragionamento: {extraction.reasoning}")
        print(f"   -> 🎯 Intento: {extraction.intent}")
        print(f"   -> 🔑 Entità: {extraction.entities}")
        print(f"   -> ⚙️  Operazioni: {ops}")
        print(f"   -> 🗂️  Filtri: {filtri}")
        
        # Salviamo tutto nello stato usando la sintassi originale a stringa
        return {
            "extraction_result": extraction,
            "messages": [f"Entità: {extraction.entities} | Filtri: {filtri} | Intento: {extraction.intent}"]
        }
    except Exception as e:
        return {"error": f"Errore Extractor: {str(e)}"}

def format_schema_for_llm(schema_list: list) -> str:
    """
    Trasforma il JSON in un formato testuale M-Schema/MAC-Schema ottimizzato per l'LLM.
    """
    formatted_tables = []
    
    for tbl in schema_list:
        name = tbl.get("table_name") or tbl.get("table", "Unknown")
        desc = tbl.get("description", tbl.get("desc", ""))
        cols = tbl.get("columns", [])
        fks = tbl.get("foreign_keys", [])
        cat_vals = tbl.get("categorical_values", "")
        samples = tbl.get("column_samples", {})
        
        # Mappatura rapida delle FK
        fk_map = {}
        for fk in fks:
            from_col = fk.get("from_column")
            to_tbl = fk.get("to_table_real") or fk.get("to_table_canonical")
            to_col = fk.get("to_column")
            if from_col and to_tbl:
                target = f"{to_tbl}.{to_col}" if to_col else to_tbl
                fk_map[from_col] = target
                
        # Costruiamo la tupla di colonne con annotazioni FK ed ESEMPI
        col_tuples = []
        for col in cols:
            col_str = col
            # 1. Aggiungiamo la FK se esiste
            if col in fk_map:
                col_str += f" [FK->{fk_map[col]}]"
            
            # 2. Aggiungiamo gli esempi se esistono
            if col in samples and samples[col]:
                # Pulizia valori per evitare "a capo" accidentali che rompono il layout
                safe_samples = [str(s).replace('\n', ' ').replace('\r', '') for s in samples[col]]
                col_str += f" (Esempi: {', '.join(safe_samples)})"
                
            col_tuples.append(col_str)
                
        # Formattazione Pseudo-Markdown
        tbl_md = f"### Tabella: {name}\n"
        if desc: 
            tbl_md += f"Descrizione: {desc}\n"
        tbl_md += f"Colonne: ( {', '.join(col_tuples)} )\n"
        if cat_vals: 
            tbl_md += f"Valori Notevoli:\n{cat_vals}\n"
            
        formatted_tables.append(tbl_md)
        
    return "\n\n".join(formatted_tables)


async def run_table_selector(state: AgentState):
    """
    NODO 2: Ottimizzato per ricerca vettoriale pulita e gestione dipendenze.
    """
    print("🔍 (Table Selector) Ricerca tabelle...")
    
    extraction = state.get("extraction_result")
    
    # --- A. PREPARAZIONE QUERY VETTORIALE ---
    if extraction and extraction.entities:
        # HYBRID QUERY: Domanda originale (per il contesto semantico) + Entità (per il boost delle keyword)
        entities_str = " ".join(extraction.entities)
        vector_search_query = f"{state['user_query']} {entities_str}"
    else:
        # Fallback sulla query intera se non ci sono entità
        vector_search_query = state["user_query"]
        
    print(f"   Testo usato per Chroma: '{vector_search_query}'")

    # Formattiamo le operazioni per il prompt
    # Fix per evitare crash se extraction è None
    if extraction and extraction.operations:
         ops_str = ", ".join(extraction.operations)
    else:
         ops_str = "None (Simple SELECT)"
    
    # --- B. RETRIEVAL (Tool) ---
    try:
        # Recuperiamo un numero generoso di tabelle (es. 10-15) per avere contesto
        schema_json = search_schema_tool.invoke({"query": vector_search_query, "k": 10})
    except Exception as e:
        return {"error": f"Errore Chroma: {str(e)}"}
    
    # --- DEBUG E CONVERSIONE ---
    
    try:
        schema_list = json.loads(schema_json) if schema_json else []
        candidate_tables = [t.get("table_name") or t.get("table") for t in schema_list if t.get("table_name") or t.get("table")]
        print(f"📦 (Table Selector) Candidate tables passate all'Agente 2: {len(candidate_tables)}")
        
        # 🔥 CREIAMO IL MARKDOWN PER L'LLM
        schema_markdown = format_schema_for_llm(schema_list)
        # 🔥 DEBUG: Salviamo il markdown in un file per poterlo ispezionare!
        with open("debug_schema_markdown.md", "w", encoding="utf-8") as f:
            f.write(schema_markdown)
        print("💾 Markdown passato all'LLM salvato in 'debug_schema_markdown.md'")
        
    except Exception as _e:
        print(f"⚠️ (Table Selector) Impossibile parsare schema_json: {_e}")
        schema_markdown = schema_json # Fallback di emergenza
        schema_list = []
    
    # --- C. SELECTION (LLM) ---
    print("🧠 (Table Selector) Filtering intelligente...")
    
    selector_prompt = """
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
    "reasoning": "Step-by-step logic in English detailing why each table was chosen and how they connect via FKs to answer the user query.",
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
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", selector_prompt),
        ("human", "### EXECUTION\nQUERY: {query}\nSCHEMA:\n{schema}\n\nOutput:")
    ])
    structured_llm = llm.with_structured_output(TableSelectionResult)
    chain = prompt | structured_llm
    
    try:
        result = await chain.ainvoke({
            "schema": schema_markdown,
            "query": state["user_query"],
            #"intent": extraction.intent if extraction else "Generic",
            #"operations": ops_str
        })
        
        # 1. Prendiamo la selezione "umana" dell'LLM
        llm_selection = result.relevant_tables
        
        # 2. Applichiamo l'Auto-Filler topologico
        #    Questo aggiungerà 'ValoriInv' se l'LLM ha scelto solo 'BeniMobili' e 'TipiValoreInv'
        final_selection = expand_selection_with_graph(
            llm_selection,
            schema_json,
            root_table_real=result.central_entity
        )
        
        # 3. Calcoliamo cosa è stato aggiunto (per log/debug)
        added_tables = set(final_selection) - set(llm_selection)
        
        reasoning_log = result.reasoning
        if added_tables:
            msg_autofix = f"\n🤖 [AUTO-FIX] Il sistema ha aggiunto tabelle ponte mancanti: {list(added_tables)}"
            reasoning_log += msg_autofix
            print(msg_autofix)
         
        # Loggare il ragionamento è fondamentale per il debug
        log_msg = f"✅ Tabelle Selezionate: {final_selection}\n🤔 Ragionamento: {reasoning_log}"
        
        return {
            "selected_tables": final_selection,
            "candidate_tables_schema": schema_json, # Utile tenerlo nello stato per debug
            "messages": [log_msg]
        }
        
    except Exception as e:
         return {"error": f"Errore Selector LLM: {str(e)}"}