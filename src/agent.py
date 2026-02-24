import json
from langchain_core.prompts import ChatPromptTemplate
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
   - **Operazioni**: parole analitiche (media, massimo, conteggio, maggiore di), devono essere espresse in linguaggio SQLite (es. AVG, MAX, COUNT, SUM, WHERE, ORDER BY, GROUP BY).
Modalità di ragionamento:
Pensa passo dopo passo. Concentrati sulla logica aziendale e sui requisiti di recupero dei dati.

#####ESEMPI#####
Input: ‘Voglio che raggruppi tutti i vincoli che ci sono per ogni fabbricato’
Output: Intent="Raggruppare i vincoli per ogni fabbricato", Entità=["Vincoli", "Fabbricati"], Operazioni=["GROUP BY"]

Input: "Voglio che calcoli la media della superficie dei terreni"
Output: Intent="Calcolare media superficie terreni", Entità=["Terreni", "Superficie"], Operazioni=["AVG"]

Input: "Quanti vincoli ci sono?"
Output: Intent="Conteggio vincoli", Entità=["Vincoli"], Operazioni=["COUNT"]
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
        
        # Log di debug
        ops = extraction.operations if extraction.operations else "Nessuna"
        print(f"   -> Entità: {extraction.entities}")
        print(f"   -> Operazioni: {ops}")
        
        return {
            "extraction_result": extraction,
            "messages": [f"Entità: {extraction.entities} | Intento: {extraction.intent}"]
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
        
        # Mappatura rapida delle FK per annotare le colonne
        fk_map = {}
        for fk in fks:
            from_col = fk.get("from_column")
            to_tbl = fk.get("to_table_real") or fk.get("to_table_canonical")
            to_col = fk.get("to_column") # <-- AGGIUNTA
            
            if from_col and to_tbl:
                # Se abbiamo anche la colonna target, la aggiungiamo (Tabella.Colonna)
                target = f"{to_tbl}.{to_col}" if to_col else to_tbl
                fk_map[from_col] = target
                
        # Costruiamo la tupla di colonne con annotazioni FK
        col_tuples = []
        for col in cols:
            if col in fk_map:
                col_tuples.append(f"{col} [FK->{fk_map[col]}]")
            else:
                col_tuples.append(col)
                
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
        # Uniamo le entità. Es: "Fabbricati Vincoli"
        vector_search_query = " ".join(extraction.entities)
        print(f"   Testo usato per Chroma: '{vector_search_query}'")
    else:
        # Fallback sulla query intera se non ci sono entità
        vector_search_query = state["user_query"]

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
    Sei un Senior Data Architect specializzato in SQL.
    
    OBIETTIVO:
    Seleziona le tabelle necessarie per rispondere in modo dettagliato alla domanda dell'utente.
    
    INPUT:
    1. DOMANDA: "{query}"
    2. TABELLE CANDIDATE (recuperate via ricerca semantica):
    {schema}
    
    ISTRUZIONI CRITICHE:
    1. **Analisi Semantica**: Usa le descrizioni e i nomi delle colonne delle tabelle per capire se contengono i dati richiesti.
    2. **Analisi Relazionale (Join)**: Se selezioni una tabella che usa una Foreign Key (es. `client_id`) per collegarsi a un concetto citato nella domanda (es. "Nome Cliente"), DEVI selezionare anche la tabella riferita se è presente nella lista.
    3. **Scarta il Rumore**: Se una tabella è stata recuperata ma non c'entra nulla con la domanda (es. tabella 'Log' per una domanda di vendita), scartala.
    4. **Colonne ID**: Le colonne che iniziano per `Id` contengono SOLO codici numerici che l'utente non può interpretare. NON puoi fermarti alla colonna ID, DEVI obbligatoriamente selezionare la tabella di destinazione seguendo la `[FK->...]` per recuperare i campi descrittivi.
    
    OUTPUT:
    Restituisci la lista delle tabelle scelte e una breve spiegazione del perché (es. "Scelgo `Orders` per gli importi e `Customers` per filtrare per nome").
    """
    
    prompt = ChatPromptTemplate.from_template(selector_prompt)
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