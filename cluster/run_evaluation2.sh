#!/bin/bash -l

# Fix for the module command
if [ -f /etc/profile ]; then
    source /etc/profile
fi
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

echo "=============================================================================="
echo " 🚀 AVVIO VALUTAZIONE AUTOMATICA BATCH (vLLM + Evaluator)"
echo "=============================================================================="

#Configuring paths and environment variables 
export SCRATCH_DIR="/scratch.hpc/leonardo.vincenzi"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p $TMPDIR $HF_HOME $PIP_CACHE_DIR

export DB_PATH="$SCRATCH_DIR/mas_text2sql/cloneDefinitivoDB.db"
export CHROMA_PATH="$SCRATCH_DIR/chroma_db_data"

export VLLM_ATTENTION_BACKEND=XFORMERS 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
mkdir -p $XDG_CACHE_HOME
export VLLM_USE_UVLOOP=0

cd "$SCRATCH_DIR/mas_text2sql"

# Activating the virtual environment
echo "⚙️ Attivazione ambiente virtuale..."
source "$SCRATCH_DIR/venv/bin/activate"

# A mechanism to terminate the vLLM if the user presses Ctrl+C during the script
trap 'echo -e "\n🛑 Interruzione manuale! Arresto server vLLM..."; pkill -f "vllm.entrypoints.openai.api_server"; exit 1;' SIGINT

# Start the vLLM server in background
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "⚠️ vLLM è già in esecuzione in background, lo riutilizzo."
else
    echo "🚀 Avvio Server vLLM (Qwen 32B) in background..."
    echo "   📄 I log del server verranno scritti in: vllm_eval_server.log"
    
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --quantization awq \
        --dtype auto \
        --api-key EMPTY \
        --port 8000 \
        --gpu-memory-utilization 0.75 \
        --max-model-len 32768 \
        --disable-log-requests > vllm_eval_server.log 2>&1 &
    
    SERVER_PID=$!
    
    echo "⏳ Attesa avvio server (Timeout 300s)..."
    timeout 300 bash -c 'until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done'
    
    # check if the door responded in time
    if [ $? -ne 0 ]; then
        echo "❌ Errore: Il server vLLM non si è avviato in tempo o è andato in crash (OOM?). Controlla vllm_eval_server.log"
        kill -9 $SERVER_PID
        exit 1
    fi
fi
echo "✅ Server vLLM pronto e in ascolto!"

# echo ""
# echo "🧹 Pulizia del vecchio Vector DB e degli indici BM25..."
# rm -rf "$CHROMA_PATH"
# rm -f "bm25_index.pkl"
# rm -f "bm25_values_index.pkl"

# echo "⚙️ Avvio Ingestion (1 tabella alla volta per evitare Timeout)..."
# export INGEST_MAX_CONCURRENCY=1
# python3 ingest_vector.py --db_path "$DB_PATH" --chroma_path "$CHROMA_PATH"

# if [ $? -ne 0 ]; then
#     echo "❌ Errore critico durante l'ingestion. Interrompo la valutazione."
#     pkill -f "vllm.entrypoints.openai.api_server"
#     exit 1
# fi
# echo "✅ Ingestion completata con successo!"

echo ""
echo "🤖 Avvio script di valutazione automatica..."
echo "--------------------------------------------------"

python3 batch_evaluator2.py --db "$DB_PATH" --golden_set set_domande.txt --output_json metriche_modello2.json --output_txt report_nodo2.txt

echo ""
echo "--------------------------------------------------"
echo "🛑 Arresto server vLLM per liberare le risorse..."
pkill -f "vllm.entrypoints.openai.api_server"

echo "🎉 Finito! Valutazione terminata con successo."
echo "I risultati sono stati salvati nel file 'metriche_modello2.json'."