#!/bin/bash
# Launch an OpenAI-compatible vLLM server for the construction agents (configs/agents.yaml
# must point to the same model name and port).
#
# MODEL_NAME is required. Everything else is optional.
#
#   MODEL_NAME=Qwen/Qwen3-32B bash hallupeer/agents/agents.sh
#   MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct DEVICES=0,1 PORT=8002 bash hallupeer/agents/agents.sh
#
# Variables (defaults in brackets):
#   MODEL_NAME              HF model id or local path             [required]
#   DEVICES                 CUDA_VISIBLE_DEVICES                  [0]
#   PORT                    server port                           [8001]
#   GPU_MEMORY_UTILIZATION  [0.95]
#   MAX_MODEL_LEN           [32768]
#   ROPE_SCALING            1 = YaRN x4 (extends Qwen3 beyond 32k) [0]

set -e

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME, e.g. MODEL_NAME=Qwen/Qwen3-32B}"
DEVICES="${DEVICES:-0}"
PORT="${PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ROPE_SCALING="${ROPE_SCALING:-0}"

export CUDA_VISIBLE_DEVICES="$DEVICES"
TENSOR_PARALLEL_SIZE=$(echo "$DEVICES" | awk -F, '{print NF}')

ARGS=(
  --model "$MODEL_NAME"
  --api-key abc
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
)
if [ "$ROPE_SCALING" = "1" ]; then
  ARGS+=(--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}')
fi

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
