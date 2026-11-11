#!/usr/bin/env bash
# =============================================================================
# RefChecker  —  Task1 claim-level NLI hallucination judge
# =============================================================================
# Extractor: Mistral-7B-Claim-Extractor (always, vLLM OpenAI endpoint).
# Checker  : one of
#              alignscore  — local AlignScore (roberta-large) — no LLM server.
#              llm         — second vLLM endpoint (e.g. Qwen/Qwen3-32B).
#
# Each (CHECKER_MODE, level) combination is run twice with
# --threshold_mode precomputed:
#   1st call : if the precompute key is missing from config_baseline.yaml, score
#              train.jsonl, sweep F1-macro, write best threshold to config, exit.
#   2nd call : the key now exists; predict on test.jsonl with that threshold.
#
# Environment overrides:
#   DATASETS               space-separated list of dataset folders     (default HalluPeer_NeurIPS2024)
#   LEVELS                 space-separated list of: review | sent     (default sent)
#   SOURCE_ROOT            root containing per-dataset paper dirs      (default ${REPO_ROOT}/dataset/template_injection)
#   OUT                    output root                                 (default evaluation_results/task1/refchecker)
#   EVIDENCE               no_retrieval | BM25                         (default BM25)
#   CHECKER_MODES          space-separated: alignscore | llm           (default "alignscore llm")
#   EXTRACTOR_MODEL        extractor HF id                             (default dongyru/Mistral-7B-Claim-Extractor)
#   EXTRACTOR_PORT         extractor vLLM port                         (default 5001)
#   EXTRACTOR_API_KEY      API key for the extractor endpoint          (default abc)
#   CHECKER_MODEL          LLM checker HF id (llm mode only)           (default Qwen/Qwen3-32B)
#   CHECKER_PORT           LLM checker vLLM port (llm mode only)       (default 5002)
#   CHECKER_API_KEY        API key for the LLM-checker endpoint        (default abc)
#   EXTRACTOR_CUDA         GPU(s) for extractor server                 (default 0)
#   CHECKER_CUDA           GPU(s) for LLM checker server               (default 1)
#   ALIGNSCORE_DEVICE      device for AlignScore (alignscore mode)     (default cuda)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_DIR}/../../.." && pwd)"
PY_SCRIPT="${BASELINE_DIR}/Refchecker.py"

DATASETS="${DATASETS:-HalluPeer_NeurIPS2024 HalluPeer_ICLR2024}"           # space-separated
LEVELS="${LEVELS:-review sent}"                                # space-separated: review | sent
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/dataset/template_injection}"
OUT="${OUT:-${REPO_ROOT}/evaluation/evaluation_results/task1/refchecker}"
EVIDENCE="${EVIDENCE:-BM25}"
CHECKER_MODES="${CHECKER_MODES:-alignscore}"

EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-dongyru/Mistral-7B-Claim-Extractor}"
EXTRACTOR_PORT="${EXTRACTOR_PORT:-5001}"
EXTRACTOR_API_KEY="${EXTRACTOR_API_KEY:-abc}"
CHECKER_MODEL="${CHECKER_MODEL:-Qwen/Qwen3-32B}"
CHECKER_PORT="${CHECKER_PORT:-5002}"
CHECKER_API_KEY="${CHECKER_API_KEY:-abc}"

EXTRACTOR_CUDA="${EXTRACTOR_CUDA:-3}"
CHECKER_CUDA="${CHECKER_CUDA:-3}"
ALIGNSCORE_DEVICE="${ALIGNSCORE_DEVICE:-cuda:3}"

# ─── Runtime settings (override via env) ───────────────────────────────────────────────────
# (optionally) the LLM checker / AlignScore process. Leave headroom by capping
# vLLM at ~45% so AlignScore (~2GB) and the second model (if any) can fit.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-python3}"

export VLLM_USE_V1=1

cd "${REPO_ROOT}"

echo "============================================================"
echo "RefChecker Task1"
echo "  DATASETS        = ${DATASETS}"
echo "  LEVELS          = ${LEVELS}"
echo "  SOURCE_ROOT     = ${SOURCE_ROOT}"
echo "  OUT             = ${OUT}"
echo "  EVIDENCE        = ${EVIDENCE}"
echo "  CHECKER_MODES   = ${CHECKER_MODES}"
echo "  EXTRACTOR_MODEL = ${EXTRACTOR_MODEL}  port=${EXTRACTOR_PORT}  cuda=${EXTRACTOR_CUDA}"
echo "  CHECKER_MODEL   = ${CHECKER_MODEL}  port=${CHECKER_PORT}  cuda=${CHECKER_CUDA}"
echo "  ALIGNSCORE_DEV  = ${ALIGNSCORE_DEVICE}"
echo "============================================================"

# Decide whether the LLM-checker vLLM will be needed at all (only if any
# checker mode in CHECKER_MODES is "llm"). If yes, we spin it up once and
# leave it running for the whole sweep.
NEED_LLM_CHECKER=0
for _m in ${CHECKER_MODES}; do
    if [[ "$_m" == "llm" ]]; then NEED_LLM_CHECKER=1; fi
done

# Warn if everything is being crammed onto a single GPU.
if [[ "$NEED_LLM_CHECKER" == "1" && "$EXTRACTOR_CUDA" == "$CHECKER_CUDA" ]]; then
    echo
    echo "[WARN] Extractor and LLM checker share CUDA=${EXTRACTOR_CUDA}. On a"
    echo "       single 80GB GPU this is likely to OOM with Qwen3-32B."
    echo "       Consider CHECKER_MODES=alignscore only, a smaller checker model,"
    echo "       or a second GPU via CHECKER_CUDA."
    echo
fi

# ─── vLLM lifecycle ──────────────────────────────────────────────────────────
EXTRACTOR_PID=""
CHECKER_PID=""

kill_one () {
    local pid="$1"; local name="$2"
    if [ -n "$pid" ]; then
        echo "Stopping ${name} (PID ${pid}) and children..."
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup () {
    kill_one "$EXTRACTOR_PID" "extractor vLLM"
    kill_one "$CHECKER_PID"   "checker vLLM"
    EXTRACTOR_PID=""
    CHECKER_PID=""
}
trap cleanup EXIT

wait_for_port () {
    local port="$1"; local name="$2"
    echo "Waiting for ${name} on port ${port}..."
    until curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
    done
    echo "${name} ready on port ${port}"
}

rope_args_for () {
    local model="$1"
    if [[ "$model" == Qwen* ]]; then
        echo '--rope-scaling {"type":"dynamic","factor":4.0}'
    fi
}

start_vllm () {
    # $1=model $2=port $3=cuda $4=log $5=name $6=api_key
    local model="$1"; local port="$2"; local cuda="$3"; local log="$4"; local name="$5"
    local api_key="$6"
    local tp
    tp=$(echo "$cuda" | awk -F, '{print NF}')
    local rope
    rope="$(rope_args_for "$model")"

    echo
    echo "------------------------------------------------------------"
    echo "Starting vLLM ${name}: ${model}  port=${port}  cuda=${cuda}"
    echo "  log=${log}"
    echo "------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES="${cuda}" "${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --api-key "${api_key}" \
        --tensor-parallel-size "${tp}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        ${rope} \
        > "${log}" 2>&1 &
    echo $!
}

# ─── Helper: run both precompute + test predict for one (dataset, level, mode) ─
run_regime () {
    local dataset="$1"; local level="$2"; local mode="$3"
    local INPUT="${REPO_ROOT}/dataset/task_split/${dataset}/task1_${level}/test.jsonl"
    local SOURCE="${SOURCE_ROOT}/${dataset}"

    local checker_args=()
    if [[ "$mode" == "llm" ]]; then
        checker_args=(
            --checker_model    "openai/${CHECKER_MODEL}"
            --checker_port     "${CHECKER_PORT}"
            --checker_api_key  "${CHECKER_API_KEY}"
        )
    else
        checker_args=( --device "${ALIGNSCORE_DEVICE}" )
    fi

    echo
    echo "============================================================"
    echo "[dataset=${dataset}  level=${level}  mode=${mode}]"
    echo "  INPUT  = ${INPUT}"
    echo "  SOURCE = ${SOURCE}"
    echo "============================================================"

    echo
    echo "  pass 1/2 (precompute threshold from train.jsonl if needed)"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --input          "${INPUT}" \
        --source_data    "${SOURCE}" \
        --out            "${OUT}" \
        --evidence       "${EVIDENCE}" \
        --checker_mode   "${mode}" \
        --extractor_model   "openai/${EXTRACTOR_MODEL}" \
        --extractor_port    "${EXTRACTOR_PORT}" \
        --extractor_api_key "${EXTRACTOR_API_KEY}" \
        "${checker_args[@]}" \
        --threshold_mode precomputed \
        --resume

    echo
    echo "  pass 2/2 (test prediction with precomputed threshold)"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --input          "${INPUT}" \
        --source_data    "${SOURCE}" \
        --out            "${OUT}" \
        --evidence       "${EVIDENCE}" \
        --checker_mode   "${mode}" \
        --extractor_model   "openai/${EXTRACTOR_MODEL}" \
        --extractor_port    "${EXTRACTOR_PORT}" \
        --extractor_api_key "${EXTRACTOR_API_KEY}" \
        "${checker_args[@]}" \
        --threshold_mode precomputed \
        --resume
}

# ─── Spin up vLLM servers ONCE for the whole sweep ──────────────────────────
EXT_LOG="${SCRIPT_DIR}/$(basename "$0" .sh)_extractor_${EXTRACTOR_MODEL//\//_}_cuda${EXTRACTOR_CUDA}_port${EXTRACTOR_PORT}.log"
EXTRACTOR_PID="$(start_vllm "${EXTRACTOR_MODEL}" "${EXTRACTOR_PORT}" "${EXTRACTOR_CUDA}" "${EXT_LOG}" "extractor" "${EXTRACTOR_API_KEY}")"
wait_for_port "${EXTRACTOR_PORT}" "extractor"

if [[ "$NEED_LLM_CHECKER" == "1" ]]; then
    CHK_LOG="${SCRIPT_DIR}/$(basename "$0" .sh)_checker_${CHECKER_MODEL//\//_}_cuda${CHECKER_CUDA}_port${CHECKER_PORT}.log"
    CHECKER_PID="$(start_vllm "${CHECKER_MODEL}" "${CHECKER_PORT}" "${CHECKER_CUDA}" "${CHK_LOG}" "checker" "${CHECKER_API_KEY}")"
    wait_for_port "${CHECKER_PORT}" "checker"
fi

# ─── Main sweep: DATASETS × LEVELS × CHECKER_MODES ──────────────────────────
for DATASET in ${DATASETS}; do
    for LEVEL in ${LEVELS}; do
        for MODE in ${CHECKER_MODES}; do
            if [[ "$MODE" != "alignscore" && "$MODE" != "llm" ]]; then
                echo "Skipping unknown CHECKER_MODE: ${MODE}" >&2
                continue
            fi
            run_regime "${DATASET}" "${LEVEL}" "${MODE}"
        done
    done
done

# vLLM servers torn down by EXIT trap.
echo
echo "============================================================"
echo "All passes done."
echo "Per-regime sweep stats:  ${OUT}/precompute/precompute_threshold_chker*.json"
echo "Saved thresholds in config: ${BASELINE_DIR}/config_baseline.yaml"
echo "Test predictions:        ${OUT}/<dataset>/task1_{level}/*.jsonl"
echo "  DATASETS = ${DATASETS}"
echo "  LEVELS   = ${LEVELS}"
echo "============================================================"
