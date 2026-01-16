import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings

# Percorso FISSO dove hai i file locali
LOCAL_MODEL_PATH = "/scratch.hpc/leonardo.vincenzi/mas_text2sql/local_models/bge-m3"

def get_shared_embedding_function():
    """
    Restituisce l'oggetto per LANGCHAIN (Chat, Agent).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔌 Factory: Caricamento modello LOCALE da {LOCAL_MODEL_PATH}")
    
    safetensor_file = os.path.join(LOCAL_MODEL_PATH, "model.safetensors")
    if not os.path.exists(safetensor_file):
        raise FileNotFoundError(f"❌ CRITICO: Manca {safetensor_file}! Riesegui la conversione.")

    return HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_PATH,
        model_kwargs={
            'device': device,
            'model_kwargs': { 
                'use_safetensors': True,
                'local_files_only': True
            }
        },
        encode_kwargs={'normalize_embeddings': True}
    )

# --- ADAPTER PER CHROMA (Fondamentale per l'ingestion) ---
class ChromaDBAdapter:
    def __init__(self, langchain_embeddings):
        self.ef = langchain_embeddings

    def __call__(self, input):
        # Converte la chiamata di Chroma in quella di LangChain
        return self.ef.embed_documents(input)
    
    def name(self):
        # Soddisfa il check di validazione di Chroma che ti dava errore
        return "bge-m3-adapter"

def get_chroma_embedding_function():
    """
    Restituisce l'oggetto adattato per CHROMA (Ingestion).
    """
    langchain_emb = get_shared_embedding_function()
    return ChromaDBAdapter(langchain_emb)