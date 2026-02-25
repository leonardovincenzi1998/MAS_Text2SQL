#!/bin/bash -l

# fix for the module command
# forces the loading of the modules environment if not present
if [ -f /etc/profile ]; then
    source /etc/profile
fi

# sometimes required specifically on CINECA:
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
# ==============================================================================
# interactive session script (Giano Cluster)
# to be launched ONLY after obtaining a node with srun
# ==============================================================================

# 1. environment setup (identical to .sbatch)
#echo "🔧 Caricamento Moduli..."
#module purge
#module load cuda
#module load gcc

# path configuration
export SCRATCH_DIR="/scratch.hpc/leonardo.vincenzi"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p $TMPDIR $HF_HOME $PIP_CACHE_DIR

# project configuration
export DB_PATH="$SCRATCH_DIR/mas_text2sql/cloneDefinitivoDB.db"
export CHROMA_PATH="$SCRATCH_DIR/chroma_db_data"

# vLLM & PyTorch stability variables
export VLLM_ATTENTION_BACKEND=XFORMERS 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
mkdir -p $XDG_CACHE_HOME
export VLLM_USE_UVLOOP=0

# in interactive mode, we usually DO NOT want to recreate the DB every time to speed up
# if needed, change to "true" or manually delete the chroma_db_data folder
export FORCE_REINGEST="false" 
# enable graph debug (0/1)
export GRAPH_DEBUG=1

# 2. activate venv
source "$SCRATCH_DIR/venv/bin/activate"

# 3. start vLLM server in background
# check if vLLM is already active to avoid port 8000 conflicts
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "⚠️ vLLM sembra già in esecuzione. Utilizzo quello esistente..."
else
    echo "🚀 Avvio Server vLLM (Qwen 32B) in background..."
    echo "   📄 I log del server verranno scritti in: vllm_server.log"
    
    # start with the SAME parameters as .sbatch
    # redirect stdout and stderr to log file to keep chat clean
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
    # wait loop
    timeout 300 bash -c 'until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done'
fi

echo "✅ Server vLLM pronto!"

# 4. ingestion management (Vector DB)
# identical logic to .sbatch: if DB is missing, it creates it.
if [ "$FORCE_REINGEST" == "true" ]; then
    echo "⚠️ FORCE_REINGEST attivo: Rimuovo vecchio DB..."
    rm -rf "$CHROMA_PATH"
fi

if [ ! -d "$CHROMA_PATH" ]; then
    echo "⚙️ Vector DB non trovato. Avvio Ingestion..."
    
    if [ ! -f "$DB_PATH" ]; then
        echo "❌ ERRORE CRITICO: Il file DB $DB_PATH non esiste!"
        # if in interactive, do not kill everything immediately, give info
        echo "Controlla il percorso e riprova."
        exit 1
    fi

    # launch ingestion
    python3 ingest_vector.py --db_path "$DB_PATH" --chroma_path "$CHROMA_PATH"
    echo "✅ Ingestion completata."
else
    echo "✅ Vector DB trovato. Salto ingestion."
fi

# 5. start interactive interface
echo ""
echo "🤖 Avvio Chat Interattiva..."
echo "--------------------------------------------------"

# intercept Ctrl+C to prevent accidental shutdown of vLLM server
trap 'echo -e "\n⚠️ Ctrl+C intercettato dal sistema. Rispondi al prompt qui sotto per uscire."' SIGINT

# infinite loop for interactive sessions, allowing multiple restarts of the Python agent without killing the vLLM server
while true; do
    LOG_FILE="chat_log_$(date +%Y-%m-%d_%H%M).txt"
    echo "📝 La conversazione verrà salvata in: $LOG_FILE"

    # launch the interactive script
    python3 interactive_main.py \
        --db "$DB_PATH" \
        --chroma_path "$CHROMA_PATH"

    echo ""
    echo "⚠️ L'agente Python è stato terminato."
    
    # under-loop for manage dirty input or accidental Ctrl+C
ì    while true; do
        read -p "🔄 Vuoi riavviare solo l'agente (es. hai modificato il codice Python)? (s/n): " restart_choice
        
        if [[ "$restart_choice" == "s" || "$restart_choice" == "S" ]]; then
            echo "⚡ Riavvio istantaneo (il server vLLM è già caldo)..."
            break # Quit from this inner loop and restart the interactive_main.py
        elif [[ "$restart_choice" == "n" || "$restart_choice" == "N" ]]; then
            break 2 # Quit from both loops and proceed to cleanup
        fi
        # if input is invalid, it will ask again without doing anything
    done
done

# restore default Ctrl+C behavior for the rest of the script (cleanup)
trap - SIGINT

# 6. cleanup
# executed only if user explicitly chose 'n' to not restart the agent, meaning they want to end the session

echo ""
echo "🛑 Arresto server vLLM..."
pkill -f "vllm.entrypoints.openai.api_server"
echo "👋 Sessione terminata."