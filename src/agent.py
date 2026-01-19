from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.graph_utils import expand_selection_with_graph
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
ENTITY_EXTRACTOR_SYSTEM_PROMPT = """
Ruolo:
Sei un esperto di comprensione delle query e recupero delle informazioni.
Sei specializzato nell'analisi delle domande degli utenti per estrarre informazioni significative per i sistemi di retrieval.
Il tuo obiettivo:
Identificare le componenti chiave da recuperare dalla domanda di un utente, compreso l'intento, le entità importanti e le operazioni rilevanti.
Il tuo compito:
1. **Rilevamento dell'intento**: determinare ciò che l'utente sta realmente chiedendo. Riassumerlo in una frase concisa.
2. **Estrazione delle parole chiave**: identificare parole chiave o frasi importanti. Trattare le entità denominate composte da più parole come singole parole chiave.
3. **Classificazione delle parole chiave**:
   - **Entità**: elementi tangibili, oggetti aziendali, nomi di tabelle, valori specifici.
   - **Operazioni**: parole analitiche (media, massimo, conteggio, maggiore di).
Modalità di ragionamento:
Pensa passo dopo passo. Concentrati sulla logica aziendale e sui requisiti di recupero dei dati.

#####ESEMPI#####
Input: ‘Voglio che raggruppi tutti i vincoli che ci sono per ogni fabbricato’
Output: Intent="Raggruppare i vincoli per ogni fabbricato", Entità=["Vincoli", "Fabbricati"], Operazioni=["Group"]

Input: "Voglio che calcoli la media della superficie dei terreni"
Output: Intent="Calcolare media superficie terreni", Entità=["Terreni", "Superficie"], Operazioni=["Average"]

Input: "Quanti vincoli ci sono?"
Output: Intent="Conteggio vincoli", Entità=["Vincoli"], Operazioni=["Conta"]
"""

# ---------------------------------------------------------
# 3. FUNZIONI DEI NODI (Logic del Grafo)
# ---------------------------------------------------------

async def run_entity_extractor(state: AgentState):
    """
    NODO 1: Nessuna modifica sostanziale necessaria, è ben fatto.
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
        return {
            "extraction_result": extraction,
            "messages": [f"Entità: {extraction.entities} | Intento: {extraction.intent}"]
        }
    except Exception as e:
        return {"error": f"Errore Extractor: {str(e)}"}


async def run_table_selector(state: AgentState):
    """
    NODO 2: Ottimizzato per ricerca vettoriale pulita e gestione dipendenze.
    """
    print("🔍 (Table Selector) Ricerca tabelle...")
    
    extraction = state.get("extraction_result")
    
    # --- MIGLIORIA 1: Query Pulita per Chroma ---
    # Usiamo SOLO le entità per la ricerca vettoriale. 
    # Le parole come "Media", "Conta", "Raggruppa" confondono il vector store.
    if extraction and extraction.entities:
        # Uniamo le entità. Es: "Fabbricati Vincoli"
        vector_search_query = " ".join(extraction.entities)
        print(f"   Testo usato per Chroma: '{vector_search_query}'")
    else:
        # Fallback sulla query intera se non ci sono entità
        vector_search_query = state["user_query"]

    # --- A. RETRIEVAL (Tool) ---
    try:
        # Recuperiamo un numero generoso di tabelle (es. 10-15) per avere contesto
        schema_json = search_schema_tool.invoke({"query": vector_search_query, "k": 10})
    except Exception as e:
        return {"error": f"Errore Chroma: {str(e)}"}
    
    # --- B. SELECTION (LLM) ---
    print("🧠 (Table Selector) Filtering intelligente...")
    
    selector_prompt = """
    Sei un Senior Data Architect specializzato in SQL.
    
    OBIETTIVO:
    Selezionare almeno il sottoinsieme minimo di tabelle necessario per rispondere alla domanda dell'utente.
    
    INPUT:
    1. DOMANDA: "{query}"
    2. TABELLE CANDIDATE (recuperate via ricerca semantica):
    {schema}
    
    ISTRUZIONI CRITICHE:
    1. **Analisi Semantica**: Usa le descrizioni delle tabelle per capire se contengono i dati richiesti.
    2. **Analisi Relazionale (Join)**: Se selezioni una tabella che usa una Foreign Key (es. `client_id`) per collegarsi a un concetto citato nella domanda (es. "Nome Cliente"), DEVI selezionare anche la tabella riferita se è presente nella lista.
    3. **Scarta il Rumore**: Se una tabella è stata recuperata ma non c'entra nulla con la domanda (es. tabella 'Log' per una domanda di vendita), scartala.
    
    OUTPUT:
    Restituisci la lista delle tabelle scelte e una breve spiegazione del perché (es. "Scelgo `Orders` per gli importi e `Customers` per filtrare per nome").
    """
    
    prompt = ChatPromptTemplate.from_template(selector_prompt)
    structured_llm = llm.with_structured_output(TableSelectionResult)
    chain = prompt | structured_llm
    
    try:
        result = await chain.ainvoke({
            "schema": schema_json,
            "query": state["user_query"],
        })
        
        # 1. Prendiamo la selezione "umana" dell'LLM
        llm_selection = result.relevant_tables
        
        # 2. Applichiamo l'Auto-Filler topologico
        #    Questo aggiungerà 'ValoriInv' se l'LLM ha scelto solo 'BeniMobili' e 'TipiValoreInv'
        final_selection = expand_selection_with_graph(llm_selection, schema_json)
        
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