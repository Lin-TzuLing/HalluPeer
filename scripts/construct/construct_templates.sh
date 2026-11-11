#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

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

# ─── Config ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0
PORT=8001
MODEL_NAME="Qwen/Qwen3-32B"

DATASETS=(
    "../datasets/openreview_data/NeurIPS2021"
    "../datasets/openreview_data/NeurIPS2022"
    "../datasets/openreview_data/NeurIPS2023"
    "../datasets/openreview_data/NeurIPS2024"
    "../datasets/openreview_data/ICLR2019"
    "../datasets/openreview_data/ICLR2020"
    "../datasets/openreview_data/ICLR2021"
    "../datasets/openreview_data/ICLR2022"
    "../datasets/openreview_data/ICLR2023"
    "../datasets/openreview_data/ICLR2024"
)

MAX_PAPERS=1200
TEMPLATE_SAMPLE_SIZE=""      # leave empty for no sampling
RESUME="--resume"            # set to "" to disable resume

PROMPT_TEMPLATE_PATH="template_construction/template_construction.txt"           # leave empty to use cfg default
PREFILTER_PROMPT_TEMPLATE_PATH="template_construction/taxonomy_prefilter.txt" # leave empty to use cfg default; set to " " to disable prefilter
OUTPUT_PATH="results/template_construction_sample{template_sample_size}/hallucination_templates_{venue}.jsonl"                    # leave empty to use cfg default
OUTPUT_USAGE_PATH="results/template_construction_sample{template_sample_size}/logs/hallucination_template_usage_{venue}.json"              # leave empty to use cfg default
BATCH_SIZE_FEASIBLE_CHECK="32"     # leave empty to use cfg default (Stage-2 semaphore)
BATCH_SIZE_PREFILTER="32"          # leave empty to use cfg default (Stage-1 semaphore)
# ──────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES
export VLLM_USE_V1=1

TENSOR_PARALLEL_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=32768

wait_for_vllm() {
    echo "Waiting for vLLM on port $PORT..."
    until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
    done
    echo "vLLM ready on port $PORT"
}

# Start vLLM once
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --api-key abc \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --port $PORT \
  --max-model-len $MAX_MODEL_LEN \
  > "$SCRIPT_DIR/$(basename "$0" .sh)_cuda${CUDA_VISIBLE_DEVICES}_port${PORT}.log" 2>&1 &

VLLM_PID=$!
wait_for_vllm

# Run construct_template for each dataset
for DATASET_DIR in "${DATASETS[@]}"; do
    echo "=== Processing dataset: $DATASET_DIR ==="

    MAX_PAPERS_ARG=""
    if [ -n "$MAX_PAPERS" ]; then
        MAX_PAPERS_ARG="--max_papers $MAX_PAPERS"
    fi

    SAMPLE_ARG=""
    if [ -n "$TEMPLATE_SAMPLE_SIZE" ]; then
        SAMPLE_ARG="--template_sample_size $TEMPLATE_SAMPLE_SIZE"
    fi

    PROMPT_ARG=""
    if [ -n "$PROMPT_TEMPLATE_PATH" ]; then
        PROMPT_ARG="--prompt_template_path $PROMPT_TEMPLATE_PATH"
    fi

    PREFILTER_ARG=""
    if [ -n "$PREFILTER_PROMPT_TEMPLATE_PATH" ]; then
        PREFILTER_ARG="--prefilter_prompt_template_path $PREFILTER_PROMPT_TEMPLATE_PATH"
    fi

    OUTPUT_ARG=""
    if [ -n "$OUTPUT_PATH" ]; then
        OUTPUT_ARG="--output_path $OUTPUT_PATH"
    fi

    OUTPUT_USAGE_ARG=""
    if [ -n "$OUTPUT_USAGE_PATH" ]; then
        OUTPUT_USAGE_ARG="--output_usage_path $OUTPUT_USAGE_PATH"
    fi

    BATCH_FC_ARG=""
    if [ -n "$BATCH_SIZE_FEASIBLE_CHECK" ]; then
        BATCH_FC_ARG="--batch_size_feasible_check $BATCH_SIZE_FEASIBLE_CHECK"
    fi

    BATCH_PF_ARG=""
    if [ -n "$BATCH_SIZE_PREFILTER" ]; then
        BATCH_PF_ARG="--batch_size_prefilter $BATCH_SIZE_PREFILTER"
    fi

    python3 -m hallupeer.dataset_construction.hallucination_injection.construct_template \
      --dataset_dir "$DATASET_DIR" \
      --port $PORT \
      $MAX_PAPERS_ARG \
      $SAMPLE_ARG \
      $PROMPT_ARG \
      $PREFILTER_ARG \
      $OUTPUT_ARG \
      $OUTPUT_USAGE_ARG \
      $BATCH_FC_ARG \
      $BATCH_PF_ARG \
      $RESUME

    echo "=== Done: $DATASET_DIR ==="
done

# Stop vLLM
kill_vllm
echo "vLLM stopped"
