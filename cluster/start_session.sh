#!/bin/bash -l

# --- FIX PER IL COMANDO MODULE ---
# Forza il caricamento dell'ambiente moduli se non è presente
if [ -f /etc/profile ]; then
    source /etc/profile
fi
# A volte su CINECA serve anche questo specifico:
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
# ==============================================================================
# SCRIPT PER SESSIONE INTERATTIVA (Giano Cluster)
# Da lanciare SOLO dopo aver ottenuto un nodo con srun
# ==============================================================================

# --- 1. SETUP AMBIENTE (Identico al .sbatch) ---
#echo "🔧 Caricamento Moduli..."
#module purge
#module load cuda
#module load gcc

# Configurazione Path
export SCRATCH_DIR="/scratch.hpc/leonardo.vincenzi"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p $TMPDIR $HF_HOME $PIP_CACHE_DIR

# Configurazione Progetto (Percorsi corretti dal tuo sbatch)
export DB_PATH="$SCRATCH_DIR/mas_text2sql/cloneDefinitivoDB.db"
export CHROMA_PATH="$SCRATCH_DIR/chroma_db_data"

# Variabili Stabilità vLLM & PyTorch
export VLLM_ATTENTION_BACKEND=XFORMERS 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
mkdir -p $XDG_CACHE_HOME
export VLLM_USE_UVLOOP=0
# In interattivo, di solito NON vogliamo ricreare il DB ogni volta per fare prima.
# Se serve forzarlo, cambia in "true" o cancella la cartella chroma_db_data a mano.
export FORCE_REINGEST="false" 

# --- 2. ATTIVAZIONE VENV ---
source "$SCRATCH_DIR/venv/bin/activate"

# --- 3. AVVIO SERVER VLLM (IN BACKGROUND) ---
# Controlliamo se vLLM è già attivo per evitare conflitti sulla porta 8000
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "⚠️ vLLM sembra già in esecuzione. Utilizzo quello esistente..."
else
    echo "🚀 Avvio Server vLLM (Qwen 32B) in background..."
    echo "   📄 I log del server verranno scritti in: vllm_server.log"
    
    # Avvio con gli STESSI parametri del .sbatch
    # Redirezioniamo stdout e stderr sul file di log per tenere pulita la chat
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --quantization awq \
        --dtype auto \
        --api-key EMPTY \
        --port 8000 \
        --gpu-memory-utilization 0.75 \
        --max-model-len 32768 \
        --disable-log-requests > vllm_server.log 2>&1 &
    
    SERVER_PID=$!
    
    echo "⏳ Attesa avvio server (Timeout 300s)..."
    # Loop di attesa
    timeout 300 bash -c 'until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done'
fi

echo "✅ Server vLLM pronto!"

# --- 4. GESTIONE INGESTION (Vector DB) ---
# Logica identica al .sbatch: se manca il DB, lo crea.

if [ "$FORCE_REINGEST" == "true" ]; then
    echo "⚠️ FORCE_REINGEST attivo: Rimuovo vecchio DB..."
    rm -rf "$CHROMA_PATH"
fi

if [ ! -d "$CHROMA_PATH" ]; then
    echo "⚙️ Vector DB non trovato. Avvio Ingestion..."
    
    if [ ! -f "$DB_PATH" ]; then
        echo "❌ ERRORE CRITICO: Il file DB $DB_PATH non esiste!"
        # Se siamo in interattivo, non killiamo tutto subito, diamo info
        echo "Controlla il percorso e riprova."
        exit 1
    fi

    # Lancia l'ingestion
    python3 ingest_vector.py --db_path "$DB_PATH" --chroma_path "$CHROMA_PATH"
    echo "✅ Ingestion completata."
else
    echo "✅ Vector DB trovato. Salto ingestion."
fi

# --- 5. AVVIO INTERFACCIA INTERATTIVA ---
echo ""
echo "🤖 Avvio Chat Interattiva..."
echo "--------------------------------------------------"

LOG_FILE="chat_log_$(date +%Y-%m-%d_%H%M).txt"
echo "📝 La conversazione verrà salvata in: $LOG_FILE"

# Lancia il nuovo script interactive_main.py
python3 interactive_main.py \
    --db "$DB_PATH" \
    --chroma_path "$CHROMA_PATH"

# --- 6. PULIZIA ---
# Quando esci dalla chat (Ctrl+C o 'exit'), chiudiamo il server
echo ""
echo "🛑 Arresto server vLLM..."
pkill -f "vllm.entrypoints.openai.api_server"
echo "👋 Sessione terminata."
