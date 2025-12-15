from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Importiamo la configurazione, i modelli e i tools
from src.config import BASE_URL, API_KEY, LLM_MODEL_NAME
from src.models import AgentState, ExtractionResult, TableSelectionResult
from src.tools import search_schema_tool

# ---------------------------------------------------------
# 1. CONFIGURAZIONE LLM (LangChain Adapter)
# ---------------------------------------------------------
# Usiamo le variabili di config.py per connetterci a Ollama/Cluster
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0
)

# ---------------------------------------------------------
# 2. PROMPT (La tua logica originale, adattata)
# ---------------------------------------------------------
# Ho mantenuto il tuo prompt eccellente, ma ho rimosso la parte 
# "Output Format" rigida perché ci penserà with_structured_output a forzarla.

ENTITY_EXTRACTOR_SYSTEM_PROMPT = """
Role:
You are a Query Understanding & Information Retrieval Expert.
You specialize in analyzing user questions to extract meaningful information for retrieval systems.

Your Goal:
Identify key retrieval points from a user's question, including the underlying intent, important entities, and relevant operations.

Your Task:
1. **Intent Detection**: Determine what the user is truly asking for. Summarize it in a concise sentence.
2. **Keyword Extraction**: Identify important keywords or phrases. Treat multi-word named entities as single keywords.
3. **Keyword Classification**:
   - **Entities**: tangible items, business objects, table names, specific values.
   - **Operations**: analytical words (average, highest, count, greater than).

Reasoning Mode:
Think step-by-step. Focus on business logic and data retrieval requirements.

#####EXAMPLES#####
Input: Group counties by state and calculate the average perimeter.
Output: Intent='Group counties by state and perform calculations', Entities=['counties', 'state'], Operations=['group', 'calculate', 'average']
"""

# ---------------------------------------------------------
# 3. FUNZIONI DEI NODI (Logic del Grafo)
# ---------------------------------------------------------

async def run_entity_extractor(state: AgentState):
    """
    NODO 1: Analizza la query utente usando il tuo Prompt specializzato.
    """
    print(f"   🕵️‍♀️ (Entity Extractor) Analisi query: '{state['user_query']}'")
    
    # Creiamo il template usando il TUO prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Usiamo with_structured_output per forzare il modello a rispondere ESATTAMENTE
    # come definito nella classe ExtractionResult (in models.py).
    # Questo sostituisce la necessità di parsare manualmente il JSON.
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = prompt | structured_llm
    
    try:
        # Usiamo ainvoke (asincrono) per performance migliori
        extraction = await chain.ainvoke({"input": state['user_query']})
        
        # Restituiamo l'aggiornamento dello stato
        return {
            "extraction_result": extraction,
            # Aggiungiamo un messaggio alla storia per debug
            "messages": [f"Entità estratte: {extraction.entities}"]
        }
    except Exception as e:
        return {"error": f"Errore in Entity Extractor: {str(e)}"}


async def run_table_selector(state: AgentState):
    """
    NODO 2: 
    1. Cerca tabelle simili in ChromaDB usando le keyword estratte dal Nodo 1.
    2. Usa l'LLM per filtrare e selezionare solo quelle utili.
    """
    print("   🔍 (Table Selector) Ricerca tabelle nel Vector Store...")
    
    extraction = state.get("extraction_result")
    
    # Fallback se l'estrazione è fallita
    if extraction:
        search_query = f"{extraction.intent} " + " ".join(extraction.entities)
    else:
        search_query = state["user_query"]

    # --- A. CHIAMATA AL TOOL (ChromaDB) ---
    try:
        # search_schema_tool.invoke è sincrono, ma va bene qui.
        # Restituisce una stringa JSON con le tabelle candidate.
        schema_json = search_schema_tool.invoke({"query": search_query, "k": 5})
    except Exception as e:
        return {"error": f"Errore nel recupero schema da Chroma: {str(e)}"}
    
    # --- B. RAGIONAMENTO LLM ---
    print("   🧠 (Table Selector) Ragionamento sulle tabelle trovate...")
    
    selector_prompt = """Sei un Data Engineer esperto. 
    Il tuo obiettivo è selezionare SOLO le tabelle SQL strettamente necessarie per rispondere alla domanda dell'utente.
    
    Hai a disposizione un sottoinsieme dello schema del database (in formato JSON) recuperato tramite ricerca semantica.
    
    SCHEMA TROVATO:
    {schema}
    
    DOMANDA UTENTE: {query}
    INTENTO ESTRATTO: {intent}
    
    Analizza la struttura delle tabelle fornite e restituisci la selezione finale.
    """
    
    prompt = ChatPromptTemplate.from_template(selector_prompt)
    structured_llm = llm.with_structured_output(TableSelectionResult)
    chain = prompt | structured_llm
    
    try:
        result: TableSelectionResult = await chain.ainvoke({
            "schema": schema_json,
            "query": state["user_query"],
            "intent": extraction.intent if extraction else "Generico"
        })
        
        return {
            "selected_tables": result.relevant_tables,
            "candidate_tables_schema": schema_json, 
            "messages": [f"Tabelle selezionate: {result.relevant_tables}"]
        }
    except Exception as e:
         return {"error": f"Errore nell'LLM Selector: {str(e)}"}
    













#     from pydantic_ai import Agent, RunContext
# from .models import SchemaDeps, ExtractionResult
# from .config import get_model

# # ---------------------------------------------------------
# # SYSTEM PROMPT (Basato su 'Entity Extraction Agent' - Appendix I)
# # ---------------------------------------------------------

# ENTITY_EXTRACTOR_SYSTEM_PROMPT = """
# Role:
# You are a Query Understanding & Information Retrieval Expert that only outputs valid JSON.
# You specialize in analyzing user questions to extract meaningful information for retrieval systems.

# Your Goal:
# Identify key retrieval points from a user's question, including the underlying intent, important entities, and relevant operations.

# Your Task:
# 1. **Intent Detection** (information retrieval focus): Determine what the user is truly asking for. Summarize it in a concise sentence.
# 2. **Keyword Extraction** (linguistic and domain focus): Identify important keywords or phrases from the question.
#    - Treat multi-word named entities (e.g., product names, department titles) and specific values (e.g., dates, codes) as single keywords.
# 3. **Keyword Classification**: Classify extracted keywords into:
#    - **Entities**: tangible or named items such as business objects, metrics, categories, table names, or specific data values.
#    - **Operations**: words or phrases describing analytical, comparison, aggregation, ordering, or logical relationships (e.g., "average", "highest", "greater than", "count").

# Reasoning Mode:
# Think step-by-step, combining linguistic and analytical logic. Focus on business logic and data retrieval requirements.

# Output Format:
# You must return a valid JSON object matching the ExtractionResult structure:
# {
#   "intent": "<brief description of the user's underlying goal>",
#   "entities": ["<entity1>", "<entity2>", "..."],
#   "operations": ["<operation1>", "<operation2>", "..."]
# }

# Constraints:
# - Do not provide explanations, reasoning, or any text outside the required JSON structure.
# - Provide structured JSON that can be used directly by downstream agents or systems.

# #####EXAMPLES#####
# Sample input natural language question: 
# Group counties by state and calculate the average perimeter in kilometers.

# Sample output: 
# {'intent': 'Group counties by state and perform calculations',
# 'entities': ['counties', 'state'],
# 'operations': ['group', 'calculate', 'average']}


# """

# # ---------------------------------------------------------
# # CONFIGURAZIONE AGENTE
# # ---------------------------------------------------------

# # Inizializzazione dell'Agente per l'Estrazione Entità
# # Usa SchemaDeps per mantenere la coerenza con il resto del progetto,
# # anche se questo specifico agente non interroga ancora il DB.
# entity_extractor_agent = Agent[SchemaDeps, ExtractionResult](
#     model=get_model(),
#     system_prompt=ENTITY_EXTRACTOR_SYSTEM_PROMPT,
#     retries=3
#     )

# # ---------------------------------------------------------
# # FUNZIONI NODO (Per l'integrazione nel Grafo)
# # ---------------------------------------------------------

# async def run_entity_extractor(state):
#     """
#     Esegue l'agente di estrazione entità sulla domanda dell'utente.
#     """
#     print(f"--- [Agente 1] ESECUZIONE ENTITY EXTRACTOR SU: '{state['user_query']}' ---")
    
#     # Poiché questo agente non usa tool SQL, non serve passare dipendenze complesse qui,
#     # ma manteniamo la struttura se in futuro servisse contesto (es. data corrente).
#     # Se SchemaDeps richiede argomenti obbligatori, vanno passati qui.
#     # Assumiamo per ora che deps possa essere None o un oggetto dummy per questo step puro NLP.
    
#     result = await entity_extractor_agent.run(state['user_query'])
    
#     print(f"--- [Agente 1] RISULTATO: {result.output} ---")
    
#     # Restituisce l'aggiornamento dello stato (chiave definita in models.AgentState)
#     return {"extraction_data": result.output}