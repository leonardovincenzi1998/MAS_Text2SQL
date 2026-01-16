import os
import torch
from safetensors.torch import save_file

# Percorso dove hai scaricato il .bin
MODEL_DIR = "/scratch.hpc/leonardo.vincenzi/mas_text2sql/local_models/bge-m3"
bin_file = os.path.join(MODEL_DIR, "pytorch_model.bin")
safe_file = os.path.join(MODEL_DIR, "model.safetensors")

print(f"🔄 Converto {bin_file} -> {safe_file}...")

if not os.path.exists(bin_file):
    print("❌ Errore: pytorch_model.bin non trovato! Hai fatto il download?")
    exit(1)

try:
    # Carichiamo usando torch puro (che bypassa i controlli paranoici di transformers)
    state_dict = torch.load(bin_file, map_location="cpu")
    
    # Salviamo in formato sicuro
    save_file(state_dict, safe_file)
    print("✅ Conversione riuscita! Ora hai model.safetensors.")
    
    # Opzionale: Rimuoviamo il file vecchio per risparmiare spazio e evitare confusione
    # os.remove(bin_file) 
except Exception as e:
    print(f"❌ Errore durante la conversione: {e}")