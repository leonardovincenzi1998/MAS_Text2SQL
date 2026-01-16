import argparse
import asyncio
import os
import sys
from langchain_core.messages import HumanMessage

# Configurazione default (fallback se non passato da argomenti)
DEFAULT_DB_PATH = "cloneDefinitivoDB.db" 

async def main():
    # --- CONFIGURAZIONE ARGOMENTI ---
    # Nota: Non chiediamo più la "query" qui, perché la chiederemo nel loop
    parser = argparse.ArgumentParser(description="Esegui l'agente Text-to-SQL in modalità INTERATTIVA")
    
    # Argomenti opzionali per configurazione
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Percorso del file database")
    parser.add_argument("--chroma_path", type=str, default=None, help="Percorso del Vector DB (Chroma)")
    
    args = parser.parse_args()
    db_path = args.db
    chroma_path = args.chroma_path

    # --- 0. SETUP AMBIENTE ---
    # Fondamentale per far trovare il DB vettoriale a tools.py
    if chroma_path:
        os.environ["CHROMA_PATH"] = chroma_path

    from src.graph import app  # ✅ import qui

    # --- 1. Verifica preliminare ---
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

    # --- LOOP INTERATTIVO ---
    while True:
        try:
            # Input Utente
            user_query = input("💬 Scrivi la tua domanda: ").strip()

            # Gestione Uscita
            if user_query.lower() in ["exit", "quit", "esci"]:
                print("\n👋 Chiusura sessione. A presto!")
                break
            
            if not user_query:
                continue

            print(f"\n🤖 Domanda Utente: '{user_query}'\n")

            # --- 2. Stato Iniziale del Grafo (Reset ad ogni domanda) ---
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "db_path": db_path,
                "selected_tables": [],
                "error": None
            }

            # --- 3. Esecuzione ---
            # Usiamo invoke normale (o ainvoke) aspettando il risultato
            final_state = await app.ainvoke(initial_state)

            # --- 4. Stampa Risultati (ESATTAMENTE COME IL TUO MAIN) ---
            print("-" * 50)
            
            # Gestione Errori
            if final_state.get("error"):
                print(f"❌ Errore durante l'esecuzione: {final_state['error']}")
            
            else:
                print("🚀 ESTRAZIONE E SELEZIONE COMPLETATA")
                
                # 1. Info sull'estrazione (Agente 1)
                if final_state.get("extraction_result"):
                    res = final_state['extraction_result']
                    print(f"\n📋 [Agente 1] Intento:     {res.intent}")
                    print(f"🔑 [Agente 1] Entità:      {res.entities}")
                    # Gestione sicura se operations non esiste
                    ops = getattr(res, 'operations', [])
                    print(f"⚙️  [Agente 1] Operazioni:  {ops}") 
                else:
                    print("\n⚠️ Nessun risultato di estrazione trovato.")

                # 2. Info su ChromaDB (Ricerca Vettoriale)
                print("\n📚 [Vector DB] Schema Recuperato:")
                if final_state.get("candidate_tables_schema"):
                    schema_len = len(final_state['candidate_tables_schema'])
                    print(f"   (JSON Schema trovato, lunghezza: {schema_len} caratteri)")
                else:
                    print("   Nessuno schema trovato.")

                # 3. Risultato Selezione (Agente 2)
                print("\n🎯 [Agente 2] TABELLE SELEZIONATE:")
                print(final_state.get("selected_tables", "Nessuna tabella selezionata"))
                
                # 4. SQL Generato (Agente 3)
                # Aggiungo questo pezzo mantenendo il tuo stile, altrimenti non vedi la query finale
                if final_state.get("generated_sql"):
                    print(f"\n✍️  [Agente 3] SQL GENERATO:")
                    print(final_state["generated_sql"])

                # Log/Ragionamento (Ultimo messaggio)
                # Nota: Se c'è l'Agente 3, l'ultimo messaggio è l'SQL. 
                # Se vuoi vedere il ragionamento dell'Agente 2, dobbiamo cercarlo nella history.
                messages = final_state["messages"]
                
                # Cerchiamo l'ultimo messaggio che sia un ragionamento (non l'SQL finale)
                reasoning_msg = None
                if len(messages) > 1:
                     # Di solito il ragionamento è il penultimo o quello dell'Agente 2
                     # Stampiamo l'ultimo messaggio disponibile come nel tuo script originale
                     last_msg = messages[-1]
                     content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                     print(f"\n🧠 [Log/Ragionamento Finale]:\n{content}")

            print("-" * 50)
            print("\n" + "x" * 50 + "\n") # Separatore tra una domanda e l'altra

        except KeyboardInterrupt:
            print("\n\n👋 Interruzione manuale rilevata. Uscita...")
            break
        except Exception as e:
            print(f"❌ Exception non gestita: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())