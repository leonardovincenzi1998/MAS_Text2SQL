import os
import torch
from safetensors.torch import save_file
import sys

# Aggiungi cartella root per importare il config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LOCAL_MODEL_PATH

if __name__ == "__main__":
    MODEL_DIR = LOCAL_MODEL_PATH
    bin_file = os.path.join(MODEL_DIR, "pytorch_model.bin")
    safe_file = os.path.join(MODEL_DIR, "model.safetensors")

    print(f"🔄 Converto {bin_file} -> {safe_file}...")

    if not os.path.exists(bin_file):
        print("❌ Errore: pytorch_model.bin non trovato! Hai fatto il download?")
        exit(1)

    try:
        # load using pure torch to bypass transformers strict checks
        state_dict = torch.load(bin_file, map_location="cpu")
        
        # save in safe format
        save_file(state_dict, safe_file)
        print("✅ Conversione riuscita! Ora hai model.safetensors.")
        
        # optional: remove old file to save space and avoid confusion
        # os.remove(bin_file) 
    except Exception as e:
        print(f"❌ Errore durante la conversione: {e}")