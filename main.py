# import asyncio
# import sqlite3
# import os
# from langchain_core.messages import HumanMessage
# from src.graph import app

# #Nome del DB temporaneo per il test
# DB_PATH = "test_inventory.db"

# def setup_dummy_db():
#     """Crea un database SQLite finto con alcuni dati per testare l'agente."""
#     if os.path.exists(DB_PATH):
#         os.remove(DB_PATH) # Pulisce vecchi test
        
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     #Creiamo tabelle che potrebbero confondere un LLM se non guarda lo schema
#     cursor.execute("CREATE TABLE products (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, price REAL NOT NULL, stock INTEGER DEFAULT 0);")
#     cursor.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER NOT NULL, order_date TEXT NOT NULL,FOREIGN KEY (customer_id) REFERENCES users(id));")
#     cursor.execute("CREATE TABLE order_items (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, product_id INTEGER NOT NULL, qty INTEGER NOT NULL,FOREIGN KEY (order_id) REFERENCES orders(id),FOREIGN KEY (product_id) REFERENCES products(id));")
#     cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, email TEXT UNIQUE);")
    
#     #Inseriamo dati dummy
#     cursor.execute("INSERT INTO products (name, price, stock) VALUES ('Laptop Gaming', 1200.50, 10);")
#     cursor.execute("INSERT INTO products (name, price, stock) VALUES ('Mouse Wireless', 25.00, 50);")
    
#     conn.commit()
#     conn.close()
#     print(f"✅ Database di test creato: {DB_PATH}")

# async def main():
#     #1. Prepara l'ambiente
#     setup_dummy_db()
    
#     #2. La domanda dell'utente
#     user_query = "Quali tabelle abbiamo che riguardano la gestione degli ordini?"
#     print(f"\n🤖 Domanda Utente: '{user_query}'\n")

#     #3. Stato Iniziale del Grafo
#     initial_state = {
#         "messages": [HumanMessage(content=user_query)],
#         "user_query": user_query,
#         "db_path": DB_PATH,
#         "selected_tables": [],
#         "error": None
#     }

#     #4. Esecuzione (Stream degli eventi)
#     #Usiamo 'stream' per vedere i passaggi, o 'ainvoke' per il risultato finale
#     final_state = await app.ainvoke(initial_state)

#     #5. Stampa Risultati
#     print("-" * 50)
#     if final_state.get("error"):
#         print(f"❌ Errore: {final_state['error']}")
#     else:
#         print("🎯 TABELLE SELEZIONATE DALL'AGENTE:")
#         print(final_state["selected_tables"])
#         print("\n🧠 RAGIONAMENTO:")
#         #Recuperiamo l'ultimo messaggio (il ragionamento dell'agente)
#         print(final_state["messages"][-1])
#     print("-" * 50)

# if __name__ == "__main__":
#     asyncio.run(main())


import argparse
import asyncio
import os
import sys
from langchain_core.messages import HumanMessage
from src.graph import app

# Configura qui il percorso assoluto o relativo del file SQLite
# Esempio: "/mnt/data/cluster_inventory_v2.sqlite"
REAL_DB_PATH = "C:\\Users\\lvincenzi\\Tesi\\Terreni_Fabbricati.db" 

async def main():
    # --- CONFIGURAZIONE ARGOMENTI ---
    parser = argparse.ArgumentParser(description="Esegui l'agente Text-to-SQL")
    
    # Argomento obbligatorio: La domanda
    parser.add_argument("query", type=str, help="La domanda dell'utente (tra virgolette)")
    
    # Argomento opzionale: Il DB (utile se vuoi cambiare file senza toccare il codice)
    parser.add_argument("--db", type=str, default=REAL_DB_PATH, help="Percorso del file database")
    
    args = parser.parse_args()
    user_query = args.query
    db_path = args.db

    # --- 1. Verifica preliminare ---
    if not os.path.exists(db_path):
        print(f"❌ ERRORE CRITICO: Il file database '{db_path}' non esiste.")
        print(f"Percorso assoluto cercato: {os.path.abspath(db_path)}")
        print("Verifica di aver caricato il file .db nella cartella di scratch!")
        sys.exit(1)

    print(f"✅ Connessione stabilita al DB: {db_path}")
    print(f"\n🤖 Domanda Utente: '{user_query}'\n")

    # --- 2. Stato Iniziale del Grafo ---
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "db_path": db_path,
        "selected_tables": [],
        "error": None
    }

    # --- 3. Esecuzione ---
    try:
        final_state = await app.ainvoke(initial_state)

        # --- 4. Stampa Risultati ---
        print("-" * 50)
        if final_state.get("error"):
            print(f"❌ Errore durante l'esecuzione: {final_state['error']}")
        else:
            print("🚀 ESTRAZIONE E SELEZIONE COMPLETATA")
            
            # Info sull'estrazione
            if final_state.get("extraction_result"):
                print(f"\n📋 Intento Estratto: {final_state['extraction_result'].intent}")
                print(f"🔑 Entità: {final_state['extraction_result'].entities}")

            # Info su ChromaDB
            print("\n📚 Schema Recuperato (ChromaDB):")
            if final_state.get("candidate_tables_schema"):
                # Stampiamo solo i primi 100 caratteri per non intasare la console, o tutto se vuoi debuggare
                print(f"   (JSON Schema grezzo disponibile nello stato, lunghezza: {len(final_state['candidate_tables_schema'])} chars)")
            else:
                print("   Nessuno schema trovato.")

            # Risultato finale
            print("\n🎯 TABELLE SELEZIONATE DALL'AGENTE:")
            print(final_state["selected_tables"])
            
            # Log dell'ultimo messaggio (Ragionamento)
            print("\n🧠 LOG:")
            print(final_state["messages"][-1])

        print("-" * 50)

    except Exception as e:
        print(f"❌ Exception non gestita: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())