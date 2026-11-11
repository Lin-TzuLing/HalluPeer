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
CUDA_VISIBLE_DEVICES=0,1
PORT=8004
POST_CHECK_PORT="8004"           # leave empty if no post checker

# ==============Injector Model============== #
MODEL_NAME="Qwen/Qwen3-32B"
OUTPUT_BASE_DIR=""               # leave empty to use value from template_injection.yaml
# ==============Injector Model============== #

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
MAX_CONCURRENT=80
TEMPLATE_CONSTRUCTION_RESULTS_PATH="results/template_construction_sampleNone/hallucination_templates_{venue}.jsonl"
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

# Run inject_template for each dataset
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

    POST_CHECK_ARG=""
    if [ -n "$POST_CHECK_PORT" ]; then
        POST_CHECK_ARG="--post_check_port $POST_CHECK_PORT"
    fi

    TEMPLATE_PATH_ARG=""
    if [ -n "$TEMPLATE_CONSTRUCTION_RESULTS_PATH" ]; then
        TEMPLATE_PATH_ARG="--template_construction_results_path $TEMPLATE_CONSTRUCTION_RESULTS_PATH"
    fi

    OUTPUT_BASE_DIR_ARG=""
    if [ -n "$OUTPUT_BASE_DIR" ]; then
        OUTPUT_BASE_DIR_ARG="--output_base_dir $OUTPUT_BASE_DIR"
    fi

    python3 -m hallupeer.dataset_construction.hallucination_injection.inject_template \
      --dataset_dir "$DATASET_DIR" \
      --port $PORT \
      --max_concurrent $MAX_CONCURRENT \
      --injector_model_name "$MODEL_NAME" \
      --post_checker_model_name "$MODEL_NAME" \
      $MAX_PAPERS_ARG \
      $SAMPLE_ARG \
      $POST_CHECK_ARG \
      $TEMPLATE_PATH_ARG \
      $OUTPUT_BASE_DIR_ARG \
      $RESUME

    echo "=== Done: $DATASET_DIR ==="
done

# Stop vLLM
kill_vllm
echo "vLLM stopped"
