import os
from langchain_openai import ChatOpenAI

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "cloneDefinitivoDB.db"))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db_data"))

# LLM Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", 'Qwen/Qwen2.5-32B-Instruct-AWQ')
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", 'hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4')
BASE_URL = os.getenv("BASE_URL", 'http://localhost:8000/v1')
API_KEY = os.getenv("API_KEY", 'EMPTY')

# Vector Store Configuration
COLLECTION_NAME = "langchain"
VALUE_COLLECTION_NAME = "db_values"
BM25_PATH = os.getenv("BM25_PATH", os.path.join(BASE_DIR, "bm25_index.pkl"))
BM25_VALUES_PATH = os.getenv("BM25_VALUES_PATH", os.path.join(BASE_DIR, "bm25_values_index.pkl"))

# Embeddings Configuration
DEFAULT_LOCAL_MODEL_PATH = "/scratch.hpc/leonardo.vincenzi/mas_text2sql/local_models/bge-m3"
LOCAL_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH)

llm_extractor = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0,
    max_tokens=1500,
    presence_penalty=0,
    frequency_penalty=0
)

llm_reasoning = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.1,
    max_tokens=1500,
    presence_penalty=0.0,
    frequency_penalty=0.0
)

llm_sql = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.0,
    max_tokens=1500,
    presence_penalty=0.0,
    frequency_penalty=0.0
)


