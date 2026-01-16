import os
import torch
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

LOCAL_MODEL_PATH = "/scratch.hpc/leonardo.vincenzi/mas_text2sql/local_models/bge-m3"

@lru_cache(maxsize=1)
def get_shared_embedding_function():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Model path non trovato: {LOCAL_MODEL_PATH}")

    # Nota: evita kwargs annidati; tieni tutto piatto.
    return HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_PATH,
        model_kwargs={
            "device": device,
            # se la tua versione lo supporta:
            # "local_files_only": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

class ChromaDBAdapter:
    def __init__(self, langchain_embeddings):
        self.ef = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.ef.embed_documents(input)

    def name(self):
        return "bge-m3-adapter"

@lru_cache(maxsize=1)
def get_chroma_embedding_function():
    return ChromaDBAdapter(get_shared_embedding_function())
