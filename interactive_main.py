import argparse
import asyncio
import os
import sys
import warnings
import traceback

from langchain_core.messages import HumanMessage

# suppress pydantic warnings for cleaner cli output
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

# default configuration
DEFAULT_DB_PATH = "cloneDefinitivoDB.db" 

# main entry point for the interactive text-to-sql cli
async def main():
    # argument parsing
    parser = argparse.ArgumentParser(description="Run the Text-to-SQL agent in INTERACTIVE mode")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
    parser.add_argument("--chroma_path", type=str, default=None, help="Path to the Chroma Vector DB directory")
    
    args = parser.parse_args()
    db_path = args.db
    chroma_path = args.chroma_path

    # environment setup crucial for tools.py to find the correct vector db path dynamically
    if chroma_path:
        os.environ["CHROMA_PATH"] = chroma_path

    # import the compiled langgraph app after setting environment variables
    from src.graph import app 

    # pre-flight checks
    if not os.path.exists(db_path):
        print(f"❌ ERRORE CRITICO: Il file database '{db_path}' non esiste.")
        print(f"Percorso assoluto cercato: {os.path.abspath(db_path)}")
        sys.exit(1)

    print("\n" + "="*60)
    print("🤖 AGENTE SQL ATTIVO - MODALITÀ INTERATTIVA")
    print(f"📁 DB Connesso: {db_path}")
    if chroma_path:
        print(f"📚 Vector DB: {chroma_path}")
    print("="*60)
    print("💡 Istruzioni: Scrivi la tua domanda e premi Invio.")
    print("   Scrivi 'exit', 'quit' o 'esci' per terminare.\n")

    # interactive loop
    while True:
        try:
            # get user input
            user_query = input("💬 Scrivi la tua domanda: ").strip()

            # handle session exit
            if user_query.lower() in ["exit", "quit", "esci"]:
                print("\n👋 Chiusura sessione. A presto!")
                break
            
            if not user_query:
                continue

            print(f"\n🤖 Domanda Utente: '{user_query}'\n")

            # initialize graph state for the current iteration
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "db_path": db_path,
                "selected_tables": [],
                "error": None
            }

            # execute the graph
            final_state = await app.ainvoke(initial_state)

            # result formatting
            print("-" * 50)
            
            # global error handling
            if final_state.get("error"):
                print(f"❌ Errore durante l'esecuzione: {final_state['error']}")
            else:
                print("🚀 ESTRAZIONE E SELEZIONE COMPLETATA")
                
                # agent 1: entity extraction stats
                if final_state.get("extraction_result"):
                    res = final_state['extraction_result']
                    ops = getattr(res, 'operations', []) 
                    print(f"\n📋 [Agente 1] Intento:     {res.intent}")
                    print(f"🔑 [Agente 1] Entità:      {res.entities}")
                    print(f"⚙️  [Agente 1] Operazioni:  {ops}") 
                else:
                    print("\n⚠️ Nessun risultato di estrazione trovato.")

                # vector db stats
                print("\n📚 [Vector DB] Schema Recuperato:")
                if final_state.get("candidate_tables_schema"):
                    schema_len = len(final_state['candidate_tables_schema'])
                    print(f"   (JSON Schema trovato, lunghezza: {schema_len} caratteri)")
                else:
                    print("   Nessuno schema trovato.")

                # agent 2: table selection result
                print("\n🎯 [Agente 2] TABELLE SELEZIONATE:")
                print(final_state.get("selected_tables", "Nessuna tabella selezionata"))
                
                # agent 3: sql generation
                if final_state.get("generated_sql"):
                    print(f"\n✍️  [Agente 3] SQL GENERATO:")
                    print(final_state["generated_sql"])

                # final reasoning log
                messages = final_state.get("messages", [])

                if len(messages) >= 2:
                    msg_agente_2 = messages[-2]
                    content_agente_2 = msg_agente_2.content if hasattr(msg_agente_2, 'content') else str(msg_agente_2)
                    print(f"\n🧠 [Ragionamento Agente 2 - Table Selector]:\n{content_agente_2}")
                    
                if len(messages) > 1:
                     last_msg = messages[-1]
                     content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                     print(f"\n🧠 [Log/Ragionamento Finale]:\n{content}")

            print("-" * 50)
            print("\n" + "x" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interruzione manuale rilevata. Uscita...")
            break
        except Exception as e:
            print(f"❌ Exception non gestita: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())