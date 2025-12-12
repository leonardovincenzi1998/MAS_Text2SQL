import os
from pydantic_ai.models.openai import OpenAIChatModel

#Configurazione
#LLM_MODEL_NAME = 'llama3.1'
#BASE_URL = 'http://localhost:11434/v1'
#API_KEY = 'ollama'

#Configurazione cluster

LLM_MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct-AWQ' 

# Configurazione vLLM su Cluster
BASE_URL = 'http://localhost:8000/v1'
API_KEY = 'EMPTY'

def get_model():
    """
    Configura il modello impostando le variabili d'ambiente.
    Questo metodo è il più sicuro perché bypassa le differenze 
    di sintassi tra le versioni della libreria.
    """
    #1. Impostiamo le variabili d'ambiente che la libreria 'openai' (usata internamente) ascolta
    os.environ['OPENAI_BASE_URL'] = BASE_URL
    os.environ['OPENAI_API_KEY'] = API_KEY

    #2. Inizializzo il modello passando SOLO il nome.
    return OpenAIChatModel(model_name=LLM_MODEL_NAME)