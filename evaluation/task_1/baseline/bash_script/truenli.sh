#!/usr/bin/env bash
# =============================================================================
# HHEM-2.1-Open  —  Task1 review-level threshold sweep
# =============================================================================
# Compares two scoring regimes:
#   (A) sentence_level=true  + agg=mean   (split review into sentences, score
#                                          each, aggregate to one truth_score)
#   (B) sentence_level=false              (entire review = one hypothesis)
#
# Each regime is run twice with --threshold_mode precomputed:
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
#   AGG                  mean | min | median                        (default mean; only used in regime A)
#   CUDA_VISIBLE_DEVICES GPU indices to expose                      (default 7)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_DIR}/../../.." && pwd)"
PY_SCRIPT="${BASELINE_DIR}/TrueNLI.py"

DATASET="${DATASET:-HalluPeer_NeurIPS2024}"
LEVEL="${LEVEL:-review sent}"                                       # space-separated list: review | sent
SOURCE="${SOURCE:-${REPO_ROOT}/dataset/template_injection/${DATASET}}"
OUT="${OUT:-${REPO_ROOT}/evaluation/evaluation_results/task1/truenli}"
EVIDENCE="${EVIDENCE:-BM25}"
AGG="${AGG:-mean}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ─── Runtime settings (override via env) ───────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES

cd "${REPO_ROOT}"

echo "============================================================"
echo "TrueNLI Task1 threshold comparison"
echo "  LEVEL    = ${LEVEL}"
echo "  SOURCE   = ${SOURCE}"
echo "  OUT      = ${OUT}"
echo "  EVIDENCE = ${EVIDENCE}"
echo "  AGG      = ${AGG}"
echo "  CUDA     = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

run_regime () {
    local label="$1"; shift
    echo
    echo "------------------------------------------------------------"
    echo "[regime ${label}]  $*"
    echo "  pass 1/2 (precompute threshold from train.jsonl if needed)"
    echo "------------------------------------------------------------"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --input         "${INPUT}" \
        --source_data   "${SOURCE}" \
        --out           "${OUT}" \
        --evidence      "${EVIDENCE}" \
        --threshold_mode precomputed \
        "$@"

    echo
    echo "  pass 2/2 (test prediction with precomputed threshold)"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --input         "${INPUT}" \
        --source_data   "${SOURCE}" \
        --out           "${OUT}" \
        --evidence      "${EVIDENCE}" \
        --threshold_mode precomputed \
        "$@"
}

for level in ${LEVEL}; do
    INPUT="${REPO_ROOT}/dataset/task_split/${DATASET}/task1_${level}/test.jsonl"
    echo
    echo "============================================================"
    echo "[level=${level}]  INPUT = ${INPUT}"
    echo "============================================================"

    # Regime A: review → sentences → score each → aggregate
    if [[ "$level" == "review" ]]; then
        run_regime "A:split+agg" \
            --sentence_level true \
            --aggregate "${AGG}"
    fi

    # Regime B: whole review as one hypothesis (no sentence split / aggregation)
    run_regime "B:whole-review" \
        --sentence_level false
done

echo
echo "============================================================"
echo "All passes done."
echo "Per-regime sweep stats:  ${OUT}/precompute/precompute_threshold_split*.json"
echo "Saved thresholds in config: ${BASELINE_DIR}/config_baseline.yaml"
echo "Test predictions:        ${OUT}/<dataset>/task1_{level}/*.jsonl  (levels: ${LEVEL})"
echo "============================================================"
