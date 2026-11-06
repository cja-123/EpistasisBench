#!/usr/bin/env bash
# Zero-shot scoring with SaProt on the EpistasisBench assays.
#
# Scoring methods supported by epistasisbench/baselines/saprot/compute_fitness.py:
#   default        mask the mutated positions, compare wild-type and mutant token probabilities
#   joint_marginal order-dependent joint marginal, falls back to Monte Carlo above depth 5
#   pll            pseudo log-likelihood, one forward pass per masked position (slow)
#   nopll          single unmasked forward pass, sum of per-position log probabilities (fast)
#
# Output: $OUT_DIR/SaProt_650M_AF2/$SAPROT_SCORING_METHOD/{DMS_id}.csv
# Existing score files are skipped, so the script can be re-run after an interruption.
# Each scoring method writes to its own directory, so switching methods does not skip assays.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../zero_shot_config.sh"

SAPROT_MODEL_PATH="${SAPROT_MODEL_PATH:?set SAPROT_MODEL_PATH to the SaProt weights}"
SAPROT_SCORING_METHOD="${SAPROT_SCORING_METHOD:-default}"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-foldseek}"               # foldseek executable, needed to build 3Di sequences

if [[ -z "$PDB_DIR" ]]; then
    echo "PDB_DIR must be set (structures used to derive 3Di sequences)" >&2
    exit 1
fi

OUT="$OUT_DIR/SaProt_650M_AF2/$SAPROT_SCORING_METHOD"
mkdir -p "$OUT"

python "$BASELINES_DIR/saprot/compute_fitness.py" \
    --foldseek_bin "$FOLDSEEK_BIN" \
    --SaProt_model_name_or_path "$SAPROT_MODEL_PATH" \
    --reference_csv "$REF_CSV" \
    --dms_dir "$DMS_DIR" \
    --structure_data_folder "$PDB_DIR" \
    --output_dir "$OUT" \
    --scoring_method "$SAPROT_SCORING_METHOD"
