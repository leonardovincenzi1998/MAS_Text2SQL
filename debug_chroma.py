import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- PERCORSO FORZATO (Copiato dal tuo messaggio) ---
CHROMA_PATH ="C:\\Users\\lvincenzi\\Tesi\\mas_text2sql\\chroma_db_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def debug_db():
    print("="*50)
    print("🕵️‍♂️ DIAGNOSTICA CHROMADB")
    print("="*50)

    # 1. VERIFICA FILESYSTEM
    print(f"\n1️⃣ Controllo cartella: {CHROMA_PATH}")
    if not os.path.exists(CHROMA_PATH):
        print("❌ ERRORE CRITICO: La cartella non esiste!")
        return
    
    files = os.listdir(CHROMA_PATH)
    print(f"   ✅ Cartella trovata. Contenuto ({len(files)} files):")
    print(f"   {files}")

    if "chroma.sqlite3" not in files:
        print("   ⚠️ ATTENZIONE: Manca 'chroma.sqlite3'. Il DB potrebbe essere vecchio o corrotto.")
    
    # 2. CARICAMENTO EMBEDDING
    print(f"\n2️⃣ Caricamento Modello Embedding ({EMBEDDING_MODEL})...")
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("   ✅ Modello caricato.")
    except Exception as e:
        print(f"   ❌ Errore caricamento modello: {e}")
        return

    # 3. CONNESSIONE DB
    print("\n3️⃣ Connessione a ChromaDB...")
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )
        # Hack per contare gli elementi senza fare query
        count = vectorstore._collection.count() 
        print(f"   ✅ Connesso. Elementi nella collezione: {count}")
        
        if count == 0:
            print("   ❌ IL DB È VUOTO! Devi rieseguire populate_chroma.py")
            return

    except Exception as e:
        print(f"   ❌ Errore connessione DB: {e}")
        return

    # 4. TEST DI RICERCA
    test_query = "fabbricati"
    print(f"\n4️⃣ Test ricerca semantica per: '{test_query}'")
    results = vectorstore.similarity_search(test_query, k=3)
    
    if not results:
        print("   ❌ NESSUN RISULTATO TROVATO (Nonostante il DB non sia vuoto).")
        print("   Possibile causa: Il modello di embedding usato ora è diverso da quello usato per creare il DB.")
    else:
        print(f"   ✅ Successo! Trovati {len(results)} risultati:")
        for doc in results:
            print(f"      - Metadata Table: {doc.metadata.get('table_name', 'N/A')}")

if __name__ == "__main__":
    debug_db()