#!/bin/bash -l

# Fix for the module command
# Forces the loading of the modules environment if not present
if [ -f /etc/profile ]; then
    source /etc/profile
fi

# Sometimes required specifically on CINECA:
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
# ==============================================================================
# Interactive session script (Giano Cluster)
# To be launched ONLY after obtaining a node with srun
# ==============================================================================

# 1. Environment setup (identical to .sbatch)
# echo "🔧 Caricamento Moduli..."
# module purge
# module load cuda
# module load gcc

# Path configuration
export SCRATCH_DIR="/scratch.hpc/leonardo.vincenzi"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p $TMPDIR $HF_HOME $PIP_CACHE_DIR

# Project configuration
export DB_PATH="$SCRATCH_DIR/mas_text2sql/cloneDefinitivoDB.db"
export CHROMA_PATH="$SCRATCH_DIR/chroma_db_data"

# vLLM & PyTorch stability variables
export VLLM_ATTENTION_BACKEND=XFORMERS 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
mkdir -p $XDG_CACHE_HOME
export VLLM_USE_UVLOOP=0

# In interactive mode, we usually DO NOT want to recreate the DB every time to speed up
# If needed, change to "true" or manually delete the chroma_db_data folder
export FORCE_REINGEST="false" 

# Enable graph debug (0/1)
export GRAPH_DEBUG=1

# 2. Activate venv
source "$SCRATCH_DIR/venv/bin/activate"

# 3. Start vLLM server in background
# Check if vLLM is already active to avoid port 8000 conflicts
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "⚠️ vLLM sembra già in esecuzione. Utilizzo quello esistente..."
else
    echo "🚀 Avvio Server vLLM (Qwen 32B) in background..."
    #echo "🚀 Avvio Server vLLM (Llama 3.3 70B) in background..."

    echo "   📄 I log del server verranno scritti in: vllm_server.log"
    
    # Start with the SAME parameters as .sbatch
    # Redirect stdout and stderr to log file to keep chat clean
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
    # Wait loop
    timeout 300 bash -c 'until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done'
fi

echo "✅ Server vLLM pronto!"

# 4. Ingestion management (Vector DB)
# Identical logic to .sbatch: if DB is missing, it creates it.
if [ "$FORCE_REINGEST" == "true" ]; then
    echo "⚠️ FORCE_REINGEST attivo: Rimuovo vecchio DB..."
    rm -rf "$CHROMA_PATH"
fi

if [ ! -d "$CHROMA_PATH" ]; then
    echo "⚙️ Vector DB non trovato. Avvio Ingestion..."
    
    if [ ! -f "$DB_PATH" ]; then
        echo "❌ ERRORE CRITICO: Il file DB $DB_PATH non esiste!"
        # If in interactive, do not kill everything immediately, give info
        echo "Controlla il percorso e riprova."
        exit 1
    fi

    # Launch ingestion
    python3 ingest_vector.py
    echo "✅ Ingestion completata."
else
    echo "✅ Vector DB trovato. Salto ingestion."
fi

# 5. Start interactive interface
echo ""
echo "🤖 Avvio Chat Interattiva..."
echo "--------------------------------------------------"

# Intercept Ctrl+C to prevent accidental shutdown of vLLM server
trap 'echo -e "\n⚠️ Ctrl+C intercettato dal sistema. Rispondi al prompt qui sotto per uscire."' SIGINT

# Infinite loop for interactive sessions, allowing multiple restarts of the Python agent without killing the vLLM server
while true; do
    LOG_FILE="chat_log_$(date +%Y-%m-%d_%H%M).txt"
    echo "📝 La conversazione verrà salvata in: $LOG_FILE"

    # Launch the interactive script
    python3 -u interactive_main.py 2>&1 | tee -a "$LOG_FILE"

    echo ""
    echo "⚠️ L'agente Python è stato terminato."
    
    # Under-loop to manage dirty input or accidental Ctrl+C
    while true; do
        read -p "🔄 Vuoi riavviare solo l'agente (es. hai modificato il codice Python)? (s/n): " restart_choice
        
        if [[ "$restart_choice" == "s" || "$restart_choice" == "S" ]]; then
            echo "⚡ Riavvio istantaneo (il server vLLM è già pronto)..."
            break # Quit from this inner loop and restart the interactive_main.py
        elif [[ "$restart_choice" == "n" || "$restart_choice" == "N" ]]; then
            break 2 # Quit from both loops and proceed to cleanup
        fi
        # If input is invalid, it will ask again without doing anything
    done
done

# Restore default Ctrl+C behavior for the rest of the script (cleanup)
trap - SIGINT

# 6. Cleanup
# Executed only if user explicitly chose 'n' to not restart the agent, meaning they want to end the session

echo ""
echo "🛑 Arresto server vLLM..."
pkill -f "vllm.entrypoints.openai.api_server"
echo "👋 Sessione terminata."