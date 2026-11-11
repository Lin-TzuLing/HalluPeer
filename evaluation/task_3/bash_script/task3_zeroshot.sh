#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Repo root: bash_script -> task_3 -> evaluation -> repo
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

VLLM_PID=""

kill_vllm() {
    if [ -n "$VLLM_PID" ]; then
        echo "Stopping vLLM (PID $VLLM_PID) and children..."
        pkill -TERM -P "$VLLM_PID" 2>/dev/null || true
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}

trap kill_vllm EXIT

# ─── Common (override via env or edit here) ──────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-8000}"

# ─── Runtime settings (override via env) ───────────────────────────────────────────────────
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-88000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS=(
    "Qwen/Qwen3-32B"
    "meta-llama/Llama-3.3-70B-Instruct"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "openai/gpt-oss-20b"
    "openai/gpt-oss-120b"
    "root-signals/RootSignals-Judge-Llama-70B"
)

# Venues to evaluate (override with e.g. DATASETS="HalluPeer_NeurIPS2024 HalluPeer_ICLR2024")
ALL_VENUES="HalluPeer_NeurIPS2021 HalluPeer_NeurIPS2022 HalluPeer_NeurIPS2023 HalluPeer_NeurIPS2024 HalluPeer_ICLR2019 HalluPeer_ICLR2020 HalluPeer_ICLR2021 HalluPeer_ICLR2022 HalluPeer_ICLR2023 HalluPeer_ICLR2024"
read -ra DATASETS <<< "${DATASETS:-$ALL_VENUES}"

EVIDENCE_LIST=("no_retrieval" "BM25")   # evidence type
MAX_SAMPLES=""                   # leave empty for all
OUT_ROOT="evaluation/evaluation_results/task3/zeroshot"
INPUT_ROOT="dataset/task_split"
SOURCE_ROOT="dataset/template_injection"
INPUT_SUBPATH="task3_review/test.jsonl"
RESUME="--resume"                     # --resume to skip existing results
API_KEY="abc"
# ──────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES
export VLLM_USE_V1=1


TENSOR_PARALLEL_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

wait_for_vllm() {
    echo "Waiting for vLLM on port $PORT..."
    until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
    done
    echo "vLLM ready on port $PORT"
}

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

for MODEL_NAME in "${MODELS[@]}"; do
  ROPE_SCALE=""
  ROPE_SCALE_ARG=""
  if [[ "$MODEL_NAME" == Qwen* ]]; then
      ROPE_SCALE="--rope-scaling"
      ROPE_SCALE_ARG='{"type":"dynamic", "factor": 4.0}'
  fi

  REASONING_MODEL_ARG=""
  MAX_TOKENS_ARG=""
  if [[ "$MODEL_NAME" == openai/gpt-oss* || "$MODEL_NAME" == openai/o* ]]; then
      REASONING_MODEL_ARG="--reasoning_model"
      MAX_TOKENS_ARG="--max_tokens 8192"
  fi

  if [[ "$MODEL_NAME" == openai/gpt-oss-120b ]]; then
      MAX_CONCURRENT=20
      RESUME="--resume"
  fi

  MODEL_SLUG="${MODEL_NAME//\//_}"

  $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --api-key $API_KEY \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    $ROPE_SCALE "$ROPE_SCALE_ARG" \
    > "$SCRIPT_DIR/$(basename "$0" .sh)_${MODEL_SLUG}_cuda${CUDA_VISIBLE_DEVICES}_port${PORT}.log" 2>&1 &

  VLLM_PID=$!
  wait_for_vllm

  for VENUE in "${DATASETS[@]}"; do
    INPUT="$INPUT_ROOT/$VENUE/$INPUT_SUBPATH"
    SOURCE_DATA="$SOURCE_ROOT/$VENUE"

    for EVIDENCE in "${EVIDENCE_LIST[@]}"; do

      echo "=== Processing model: $MODEL_NAME | dataset: $VENUE | evidence: $EVIDENCE ==="
      echo "    input:       $INPUT"
      echo "    source_data: $SOURCE_DATA"

      $PYTHON_BIN "$SCRIPT_DIR/../zeroshot.py" \
        --input "$INPUT" \
        --source_data "$SOURCE_DATA" \
        --out "$OUT_ROOT" \
        --evidence "$EVIDENCE" \
        --model "$MODEL_NAME" \
        --port $PORT \
        --api_key $API_KEY \
        --max_concurrent $MAX_CONCURRENT \
        $MAX_TOKENS_ARG \
        $MAX_SAMPLES_ARG \
        $RESUME \
        $REASONING_MODEL_ARG

      echo "=== Done: $MODEL_NAME | $VENUE | evidence: $EVIDENCE ==="
    done
  done

  kill_vllm
  echo "vLLM stopped for $MODEL_NAME"
  sleep 5   # short break between models
done
