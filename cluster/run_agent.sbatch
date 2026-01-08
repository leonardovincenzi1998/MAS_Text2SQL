#!/bin/bash
#SBATCH --job-name=sql_agent_rag
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=l40       # GPU L40 (48GB) perfetta per Qwen 32B
#SBATCH --gres=gpu:1
#SBATCH --mem=58G
#SBATCH --output=agent_log_%j.txt
#SBATCH --chdir=/scratch.hpc/leonardo.vincenzi/mas_text2sql

# --- 1. CONFIGURAZIONE PATH E VARIABILI ---
export SCRATCH_DIR="/scratch.hpc/leonardo.vincenzi"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p $TMPDIR $HF_HOME $PIP_CACHE_DIR

# Configurazione specifica per il progetto
export DB_PATH="$SCRATCH_DIR/mas_text2sql/cloneDefinitivoDB.db"
export CHROMA_PATH="$SCRATCH_DIR/chroma_db_data"

# VARIABILI DI STABILITÀ VLLM (BLOCCANO L'ERRORE FLASHINFER)
export VLLM_ATTENTION_BACKEND=XFORMERS     # <--- USA XFORMERS (PRECOMPILATO) INVECE DI FLASHINFER
# Variabili PyTorch/VLLM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
mkdir -p $XDG_CACHE_HOME

# VARIABILE DI CONTROLLO
export FORCE_REINGEST="true"

# --- 2. ATTIVAZIONE VENV ---
source "$SCRATCH_DIR/venv/bin/activate"

#Inserito nel file requirements_cluster.txt
#echo "🔧 FIX: Downgrade vLLM alla versione stabile per Giano..."
#pip uninstall -y vllm
# Installiamo la 0.6.4.post1 che è rock-solid per Qwen 2.5
#pip install vllm==0.6.4.post1 --no-cache-dir

# Verifica installazione chromadb (scommentare se serve installarlo al volo)
# pip install chromadb --no-cache-dir

echo "⬇️ Controllo modello Qwen..."
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ

# --- 3. AVVIO SERVER VLLM ---
echo "🚀 Avvio Server vLLM..."
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --dtype auto \
    --api-key EMPTY \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 &

SERVER_PID=$!

echo "⏳ Attesa avvio server (Timeout 300s)..."
timeout 300 bash -c 'until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done'
echo "✅ Server vLLM pronto!"

# --- 4. GESTIONE INGESTION (Vector DB) ---
# Controlliamo se il database vettoriale esiste già.
# Se NON esiste, lanciamo lo script di ingestion che usa l'LLM appena avviato.

if [ "$FORCE_REINGEST" == "false" ]; then
    echo "⚠️ FORCE_REINGEST è attivo: Rimuovo il vecchio vector DB se esiste..."
    rm -rf "$CHROMA_PATH"
fi

if [ ! -d "$CHROMA_PATH" ]; then
    echo "⚙️ Vector DB non trovato (o rimosso). Avvio Ingestion con Qwen 32B..."
    
    # IMPORTANTE: Assicurati che lo script python sappia dove si trova il DB SQL
    # Passando le variabili d'ambiente o argomenti
    export DB_PATH="$DB_PATH"
    export CHROMA_PATH="$CHROMA_PATH"
    
    # Lancia l'ingestion usando il server vLLM locale
    # (Lo script deve usare src.config che punta a localhost:8000)
    python3 ingest_vector.py --db_path "$DB_PATH" --chroma_path "$CHROMA_PATH"
    
    echo "✅ Ingestion completata."
else
    echo "✅ Vector DB trovato. Salto l'ingestion."
fi

# --- 5. ESECUZIONE AGENTE ---
echo "🤖 Avvio Agente Text-to-SQL RAG..."

# Passiamo sia il DB SQL che il path di Chroma
# (Assumi che il tuo main.py sia stato aggiornato per accettare --chroma_path o lo legga da config)
python3 main.py "Quante specie di beni mobili ci sono?" \
    --db "$DB_PATH" \
    --chroma_path "$CHROMA_PATH"

# --- 6. PULIZIA ---
echo "🛑 Arresto server vLLM..."
kill $SERVER_PID