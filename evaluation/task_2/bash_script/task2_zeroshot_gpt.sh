#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Repo root: bash_script -> task_2 -> evaluation -> repo
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS=(
    "gpt-5.2-2025-12-11"
)

# Venues to evaluate (override with e.g. DATASETS="HalluPeer_NeurIPS2024 HalluPeer_ICLR2024")
ALL_VENUES="HalluPeer_NeurIPS2021 HalluPeer_NeurIPS2022 HalluPeer_NeurIPS2023 HalluPeer_NeurIPS2024 HalluPeer_ICLR2019 HalluPeer_ICLR2020 HalluPeer_ICLR2021 HalluPeer_ICLR2022 HalluPeer_ICLR2023 HalluPeer_ICLR2024"
read -ra DATASETS <<< "${DATASETS:-$ALL_VENUES}"

EVIDENCE_LIST=("BM25")
MAX_SAMPLES=""
OUT_ROOT="evaluation/evaluation_results/task2/zeroshot"
INPUT_ROOT="dataset/task_split"
SOURCE_ROOT="dataset/template_injection"
LEVEL_LIST=("review" "sent")
RESUME="--resume"
MAX_CONCURRENT=20
PYTHON_BIN="python3"
PORT=8000
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(tr -d '[:space:]' < openai_api_key.txt)}"
# ──────────────────────────────────────────────────────────────────────────────

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

for MODEL_NAME in "${MODELS[@]}"; do
  for VENUE in "${DATASETS[@]}"; do
    SOURCE_DATA="$SOURCE_ROOT/$VENUE"

    for LEVEL in "${LEVEL_LIST[@]}"; do
      INPUT="$INPUT_ROOT/$VENUE/task2_${LEVEL}/test.jsonl"

      for EVIDENCE in "${EVIDENCE_LIST[@]}"; do

        echo "=== Processing model: $MODEL_NAME | level: $LEVEL | dataset: $VENUE | evidence: $EVIDENCE ==="
        echo "    input:       $INPUT"
        echo "    source_data: $SOURCE_DATA"

        $PYTHON_BIN "$SCRIPT_DIR/../zeroshot.py" \
          --input "$INPUT" \
          --source_data "$SOURCE_DATA" \
          --out "$OUT_ROOT" \
          --evidence "$EVIDENCE" \
          --model "$MODEL_NAME" \
          --port $PORT \
          --api_key $OPENAI_API_KEY_VAL \
          --max_concurrent $MAX_CONCURRENT \
          --max_tokens 8192 \
          --reasoning_model \
          $MAX_SAMPLES_ARG \
          $RESUME

        echo "=== Done: $MODEL_NAME | $VENUE | $LEVEL | evidence: $EVIDENCE ==="
      done
    done  # LEVEL
  done  # VENUE
done
