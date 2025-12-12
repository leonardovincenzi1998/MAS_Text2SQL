#!/bin/bash
#SBATCH --job-name=sql_agent
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=l40       # Specifica la coda con le GPU da 48GB (OBBLIGATORIA per 32B)
#SBATCH --gres=gpu:1          # Richiede 1 GPU
#SBATCH --mem=64G             # Richiede RAM di sistema adeguata
#SBATCH --output=agent_log_%j.txt
#SBATCH --chdir=/scratch.hpc/leonardo.vincenzi/mas_text2sql

# 1. Setup Variabili (Cruciale per non riempire la Home)
export HF_HOME=/scratch.hpc/leonardo.vincenzi/hf_cache
export PIP_CACHE_DIR=/scratch.hpc/leonardo.vincenzi/.cache/pip # Devia la cache pip su scratch per sicurezza
export TMPDIR=/scratch.hpc/leonardo.vincenzi/tmp # Devia i file temporanei
mkdir -p $TMPDIR

# 2. Attiva Venv
source /scratch.hpc/leonardo.vincenzi/venv/bin/activate

# 3. Avvio vLLM (Server)
# NOTA: Qwen 32B AWQ richiede circa 18-20GB di VRAM. 
# La L40 ne ha 48GB, quindi abbiamo spazio per un contesto ampio (8192).
echo "🚀 Avvio Server vLLM con Qwen 2.5 32B AWQ..."
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --dtype auto \
    --api-key EMPTY \
    --port 8000 \
    --max-model-len 8192 &

SERVER_PID=$!

# 4. Attesa Server
echo "⏳ Attesa avvio server..."
# Timeout impostato a 5 minuti (il caricamento di 20GB di pesi può richiedere 1-2 minuti)
timeout 300 bash -c 'until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done'

# 5. Esecuzione Agente
echo "✅ Server pronto! Avvio agente..."
# Modifica qui la domanda tra virgolette per testare query diverse
python3 main.py "Quanti terreni ci sono?"

# 6. Pulizia
kill $SERVER_PID