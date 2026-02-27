import os
import torch
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import LOCAL_MODEL_PATH

# initializes and caches the HuggingFace embedding model
@lru_cache(maxsize=1)
def get_shared_embedding_function() -> HuggingFaceEmbeddings:
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # Force CPU usage for gpu savings

    if not os.path.isdir(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {LOCAL_MODEL_PATH}")

    return HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_PATH,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

# adapter to make LangChain embeddings compatible with ChromaDB's expected interface
class ChromaDBAdapter:
    def __init__(self, langchain_embeddings: HuggingFaceEmbeddings):
        self.ef = langchain_embeddings

    # embeds the input documents, handling both strings and lists of strings
    def __call__(self, input) -> list:
        if isinstance(input, str):
            input = [input]
        return self.ef.embed_documents(input)

    def name(self) -> str:
        return "bge-m3-adapter"

# returns a cached instance of the ChromaDB embedding adapter
@lru_cache(maxsize=1)
def get_chroma_embedding_function() -> ChromaDBAdapter:
    return ChromaDBAdapter(get_shared_embedding_function())