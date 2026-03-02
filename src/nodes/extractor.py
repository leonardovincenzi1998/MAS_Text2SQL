from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models import AgentState, ExtractionResult
from src.config import llm_reasoning
from src.prompts import ENTITY_EXTRACTOR_SYSTEM_PROMPT

# node 1: entity, operations, and filters extraction
async def run_entity_extractor(state: AgentState) -> Dict[str, Any]:
    print(f"🕵️‍♀️ (Entity Extractor) Analisi query: '{state['user_query']}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    structured_llm = llm_reasoning.with_structured_output(ExtractionResult).with_retry(stop_after_attempt=3)
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