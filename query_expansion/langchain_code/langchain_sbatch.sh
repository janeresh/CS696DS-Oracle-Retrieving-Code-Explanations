#!/bin/bash
#SBATCH --job-name=langchain_exp
#SBATCH --output=logs/langchain_exp_%j.out
#SBATCH --error=logs/langchain_exp_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="vram40"
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source /work/pi_wenlongzhao_umass_edu/27/.venv/bin/activate

# Set static model and port (you can override with sbatch --export=PORT=8001)
MODEL="granite"
PORT="8500"  # Default to 8000 if not set

declare -A MODEL_PATHS
MODEL_PATHS["deepseek"]="/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa"
MODEL_PATHS["granite"]="/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"

MODEL_PATH="${MODEL_PATHS[$MODEL]}"

if [ -z "$MODEL_PATH" ]; then
    echo "Unknown model name: $MODEL"
    exit 1
fi

echo "Starting $MODEL on port $PORT"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL-chat" \
  --host 0.0.0.0 \
  --port "$PORT" > logs/${MODEL}_server_${PORT}.log 2>&1 &

echo "⏳ Waiting for server on port $PORT..."
until curl -s "http://0.0.0.0:$PORT/v1/models" | grep -q "$MODEL"; do
    sleep 5
done

# Pass model and port to Python script
python /work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/llm_langchain_query_exp.py "$MODEL" "$PORT"

echo "Finished on port $PORT"
