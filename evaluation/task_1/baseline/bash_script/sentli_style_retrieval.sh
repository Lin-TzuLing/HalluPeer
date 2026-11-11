#!/usr/bin/env bash
# =============================================================================
# seNtLI-style retrieval  —  Task1
# =============================================================================
# Compares two scoring modes:
#   (A) mode=topk    : top-k doc sentences by entailment → context → rerank
#   (B) mode=sentli  : top-ke by entailment ∪ top-kc by contradiction → rerank
#
# Each mode is run twice with --threshold_mode precomputed:
#   1st call : if the precompute key is missing from config_baseline.yaml, score
#              train.jsonl, sweep F1-macro, write best threshold to config, exit.
#   2nd call : the key now exists; predict on test.jsonl with that threshold.
#
# Environment overrides:
#   DATASET              dataset folder under task_split/  (default HalluPeer_NeurIPS2024)
#   INPUT                full path to test.jsonl                   (default task1_sent/test.jsonl)
#   SOURCE               root of {paper_id}.json files             (REQUIRED — set before run)
#   OUT                  output root                                (default evaluation_results/...)
#   EVIDENCE             no_retrieval | BM25                        (default BM25)
#   K                    top-k doc sentences for context            (default 3)
#   CUDA_VISIBLE_DEVICES GPU indices to expose                      (default 0,1)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_DIR}/../../.." && pwd)"
PY_SCRIPT="${BASELINE_DIR}/seNtLI-style_retrieval.py"

DATASET="${DATASET:-HalluPeer_NeurIPS2024}"
LEVEL="${LEVEL:-review sent}"                                       # space-separated list: review | sent
SOURCE="${SOURCE:-${REPO_ROOT}/dataset/template_injection/${DATASET}}"
OUT="${OUT:-${REPO_ROOT}/evaluation/evaluation_results/task1/sentli}"
EVIDENCE="${EVIDENCE:-BM25}"
K="${K:-3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ─── Runtime settings (override via env) ───────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES

cd "${REPO_ROOT}"

echo "============================================================"
echo "seNtLI Task1 mode comparison"
echo "  LEVEL    = ${LEVEL}"
echo "  SOURCE   = ${SOURCE}"
echo "  OUT      = ${OUT}"
echo "  EVIDENCE = ${EVIDENCE}"
echo "  K        = ${K}"
echo "  CUDA     = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

run_mode () {
    local label="$1"; shift
    echo
    echo "------------------------------------------------------------"
    echo "[mode ${label}]  $*"
    echo "  pass 1/2 (precompute threshold from train.jsonl if needed)"
    echo "------------------------------------------------------------"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --input          "${INPUT}" \
        --source_data    "${SOURCE}" \
        --out            "${OUT}" \
        --evidence       "${EVIDENCE}" \
        --k              "${K}" \
        --threshold_mode precomputed \
        "$@"

    echo
    echo "  pass 2/2 (test prediction with precomputed threshold)"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --input          "${INPUT}" \
        --source_data    "${SOURCE}" \
        --out            "${OUT}" \
        --evidence       "${EVIDENCE}" \
        --k              "${K}" \
        --threshold_mode precomputed \
        "$@"
}

for level in ${LEVEL}; do
    INPUT="${REPO_ROOT}/dataset/task_split/${DATASET}/task1_${level}/test.jsonl"
    echo
    echo "============================================================"
    echo "[level=${level}]  INPUT = ${INPUT}"
    echo "============================================================"

    # Mode A: top-k by entailment
    run_mode "A:topk" \
        --mode topk

    # Mode B: top-ke by entailment ∪ top-kc by contradiction
    run_mode "B:sentli" \
        --mode sentli
done

echo
echo "============================================================"
echo "All modes done."
echo "Per-mode sweep stats:  ${OUT}/precompute/precompute_threshold_mode*.json"
echo "Saved thresholds in config: ${BASELINE_DIR}/config_baseline.yaml"
echo "Test predictions:      ${OUT}/<dataset>/task1_{level}/*.jsonl  (levels: ${LEVEL})"
echo "============================================================"
