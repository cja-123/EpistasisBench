#!/usr/bin/env bash
# Zero-shot scoring with ESM3 / ESM-C on the EpistasisBench assays.
#
# Scoring methods supported by epistasisbench/baselines/esm3/compute_fitness.py:
#   default              per-mutation, mask the mutated positions only
#   joint_marginal       order-dependent joint marginal, needs ESM3_MODEL_TYPE=esm3_open
#   mask_marginal        mask all mutated positions at once
#   pseudo_loglikelihood pseudo log-likelihood of the mutant minus wild type
#
# Output: $OUT_DIR/esm3/<model_type>/{DMS_id}.csv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../zero_shot_config.sh"

ESM3_MODEL_TYPE="${ESM3_MODEL_TYPE:-esm3_open}"        # esm3_open | esmc_300M | esmc_600M
ESM3_MODEL_PATH="${ESM3_MODEL_PATH:-}"                 # optional local checkpoint dir, empty = download
ESM3_SCORING_METHOD="${ESM3_SCORING_METHOD:-default}"

case "$ESM3_MODEL_TYPE" in
    esm3_open) MODEL_LABEL="ESM3" ;;
    esmc_300M) MODEL_LABEL="ESMC-300M" ;;
    esmc_600M) MODEL_LABEL="ESMC-600M" ;;
    *)         MODEL_LABEL="$ESM3_MODEL_TYPE" ;;
esac

if [[ "$ESM3_MODEL_TYPE" == "esm3_open" && "$ESM3_SCORING_METHOD" == "joint_marginal" && -z "$PDB_DIR" ]]; then
    echo "PDB_DIR must be set for esm3_open + joint_marginal" >&2
    exit 1
fi

OUT="$OUT_DIR/$MODEL_LABEL"
mkdir -p "$OUT"

python "$BASELINES_DIR/esm3/compute_fitness.py" \
    --model_type "$ESM3_MODEL_TYPE" \
    ${ESM3_MODEL_PATH:+--model_path "$ESM3_MODEL_PATH"} \
    --reference_csv "$REF_CSV" \
    --dms_dir "$DMS_DIR" \
    --pdb_dir "$PDB_DIR" \
    --output_dir "$OUT" \
    --scoring_method "$ESM3_SCORING_METHOD" \
    --devices "$GPU" \
    --batch_size "$BATCH_SIZE"
