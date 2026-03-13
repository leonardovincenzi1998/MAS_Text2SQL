from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models import AgentState, ExtractionResult
from src.config import llm_extractor
from src.prompts import ENTITY_EXTRACTOR_SYSTEM_PROMPT

# node 1: entity, operations, and filters extraction
async def run_entity_extractor(state: AgentState) -> Dict[str, Any]:
    print(f"🕵️‍♀️ (Entity Extractor) Analisi query: '{state['user_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    structured_llm = llm_extractor.with_structured_output(ExtractionResult).with_retry(stop_after_attempt=3)
    chain = prompt | structured_llm
    
    try:
        extraction: ExtractionResult = await chain.ainvoke({"input": state['user_query']})
        
        # safely handle empty lists
        filtri = extraction.filters if extraction.filters else ["None"]
        chiavi_ricerca = extraction.search_keywords if hasattr(extraction, 'search_keywords') and extraction.search_keywords else ["None"]

        if extraction.entities:
            # Estrae la stringa leggibile: "[Categoria: Valore], [Categoria: Valore]"
            entita_formattate = ", ".join([f"[{e.category}: '{e.value}']" for e in extraction.entities])
        else:
            entita_formattate = "None"

        print(f"   -> 🎯 Intent: {extraction.intent}")
        print(f"   -> 🔑 Entities: {entita_formattate}") 
        print(f"   -> 🔎 Search Keywords: {chiavi_ricerca}")
        print(f"   -> 🗂️ Filters: {filtri}")
        
        return {
            "extraction_result": extraction,
            "messages": [f"Entities: {entita_formattate} | Filters: {filtri} | Intent: {extraction.intent} | Search Keywords: {chiavi_ricerca}"]
        }
    except Exception as e:
        print(f"   ❌ (Entity Extractor) Errore CRITICO: {str(e)}")
        return {"error": f"Extractor Error: {str(e)}"}