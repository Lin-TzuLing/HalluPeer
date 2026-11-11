#!/usr/bin/env bash
# =============================================================================
# RA-LLM  —  Task1 hallucination judge w/ BM25 retrieval
# =============================================================================
# Runs RA-LLM.py across one or more modes (kr | cot | contrast).
#
# Pipeline per mode:
#   1. Start a vLLM OpenAI-compatible server for the model.
#   2. Run RA-LLM.py: sentence split → per-sentence BM25 → LLM judge →
#      aggregate → threshold → pred.
#   3. Stop vLLM before moving to the next model.
#
# Environment overrides:
#   DATASET              dataset folder under task_split/  (default HalluPeer_NeurIPS2024)
#   INPUT                full path to test.jsonl                   (default task1_review/test.jsonl)
#   SOURCE               root of {paper_id}.json files             (REQUIRED — set before run)
#   OUT                  output root                                (default evaluation_results/task1/ra_llm)
#   EVIDENCE             no_retrieval | BM25 | LCS               (default BM25)
#   CUDA_VISIBLE_DEVICES GPU indices to expose                      (default 0)
#   PORT                 vLLM port                                  (default 8000)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_DIR}/../../.." && pwd)"
PY_SCRIPT="${BASELINE_DIR}/RA-LLM.py"


DATASET="${DATASET:-HalluPeer_NeurIPS2024}"
SOURCE="${SOURCE:-${REPO_ROOT}/dataset/template_injection/${DATASET}}"
OUT="${OUT:-${REPO_ROOT}/evaluation/evaluation_results/task1/ra_llm}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-8001}"

# Levels to run (review | sent)
LEVELS=(
    "review"
    "sent"
)

# Evidence types to run (no_retrieval | BM25 | LCS)
EVIDENCES=(
    "no_retrieval"
    "BM25"
    "LCS"
)

# Prompting modes to run
MODES=(
    "kr"
    "cot"
    "contrast"
)

# Models to run
MODELS=(
    "Qwen/Qwen3-32B"
)

# ─── Runtime settings (override via env) ───────────────────────────────────────────────────
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-88000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-15}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES
export VLLM_USE_V1=1


TENSOR_PARALLEL_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

# ─── vLLM lifecycle ──────────────────────────────────────────────────────────
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

wait_for_vllm() {
    echo "Waiting for vLLM on port ${PORT}..."
    until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
    done
    echo "vLLM ready on port ${PORT}"
}

# ─── Main loop ───────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

echo "============================================================"
echo "RA-LLM Task1"
echo "  SOURCE         = ${SOURCE}"
echo "  OUT            = ${OUT}"
echo "  LEVELS         = ${LEVELS[*]}"
echo "  EVIDENCES      = ${EVIDENCES[*]}"
echo "  MODES          = ${MODES[*]}"
echo "  MODELS         = ${MODELS[*]}"
echo "  CUDA           = ${CUDA_VISIBLE_DEVICES}"
echo "  PORT           = ${PORT}"
echo "============================================================"

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_SLUG="${MODEL_NAME//\//_}"

    ROPE_SCALE_ARGS=()
    if [[ "$MODEL_NAME" == Qwen* ]]; then
        ROPE_SCALE_ARGS=(--rope-scaling '{"type": "dynamic", "factor": 4.0}')
    fi

    REASONING_ARG=""
    MAX_TOKENS_ARG=""
    if [[ "$MODEL_NAME" == openai/gpt-oss* || "$MODEL_NAME" == openai/o* ]]; then
        REASONING_ARG="--reasoning_model"
        MAX_TOKENS_ARG="--max_tokens 8192"
    fi

    LOG="${SCRIPT_DIR}/$(basename "$0" .sh)_${MODEL_SLUG}_cuda${CUDA_VISIBLE_DEVICES}_port${PORT}.log"

    echo
    echo "------------------------------------------------------------"
    echo "Starting vLLM for ${MODEL_NAME}"
    echo "------------------------------------------------------------"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_NAME}" \
        --api-key abc \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --port "${PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        "${ROPE_SCALE_ARGS[@]+"${ROPE_SCALE_ARGS[@]}"}" \
        > "${LOG}" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm

    for LEVEL in "${LEVELS[@]}"; do
        INPUT="${REPO_ROOT}/dataset/task_split/${DATASET}/task1_${LEVEL}/test.jsonl"
        for EVIDENCE in "${EVIDENCES[@]}"; do
            for MODE in "${MODES[@]}"; do
                echo
                echo "------------------------------------------------------------"
                echo "[model ${MODEL_NAME}]  level=${LEVEL}  evidence=${EVIDENCE}  mode=${MODE}"
                echo "  INPUT = ${INPUT}"
                echo "------------------------------------------------------------"
                "${PYTHON_BIN}" "${PY_SCRIPT}" \
                    --input          "${INPUT}" \
                    --source_data    "${SOURCE}" \
                    --out            "${OUT}" \
                    --evidence       "${EVIDENCE}" \
                    --mode           "${MODE}" \
                    --model          "${MODEL_NAME}" \
                    --port           "${PORT}" \
                    --max_concurrent "${MAX_CONCURRENT}" \
                    --resume \
                    ${MAX_TOKENS_ARG} \
                    ${REASONING_ARG}
            done
        done
    done

    kill_vllm
    echo "vLLM stopped for ${MODEL_NAME}"
    sleep 5
done

echo
echo "============================================================"
echo "All done."
echo "Results: ${OUT}/<dataset>/task1_*/*.jsonl"
echo "============================================================"
