#!/usr/bin/env bash
# Zero-shot scoring with S3F on the EpistasisBench assays.
#
# Scoring methods supported by epistasisbench/baselines/s3f/compute_fitness.py:
#   masked_marginal_a     all mutated positions masked simultaneously
#   joint_marginal        order-dependent joint marginal (Monte Carlo above depth 5)
#   pseudo_loglikelihood  pseudo log-likelihood
#
# Several methods can be passed at once, e.g.
#   S3F_SCORING_METHODS="masked_marginal_a joint_marginal" ./scoring_s3f.sh
#
# This script needs the upstream S3F code, which is not part of this repository:
#   S3F_HOME/s3f/        model, dataset and protein modules
#   S3F_HOME/script/     evaluate.py
# Put them under epistasisbench/baselines/s3f/ or point S3F_HOME at the S3F checkout.
#
# The yaml config (S3F_CONFIG) must define:
#   summary      -> dataset summary object, its path must be the substitution csv directory
#   task         -> fusion network task
#   model_checkpoint -> trained S3F weights
#   structure_path, surface_path -> PDB and surface graphs
#
# Output: $OUT_DIR/s3f/{DMS_filename}

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../zero_shot_config.sh"

S3F_CONFIG="${S3F_CONFIG:?set S3F_CONFIG to the S3F yaml configuration}"
S3F_MODEL_NAME="${S3F_MODEL_NAME:-S3F}"                  # column prefix in the output csv
S3F_SCORING_METHODS="${S3F_SCORING_METHODS:-masked_marginal_a}"
S3F_HOME="${S3F_HOME:-$BASELINES_DIR/s3f}"

if [[ ! -f "$S3F_HOME/s3f/model.py" || ! -f "$S3F_HOME/script/evaluate.py" ]]; then
    echo "S3F code not found under $S3F_HOME (expected s3f/model.py and script/evaluate.py)" >&2
    exit 1
fi

OUT="$OUT_DIR/S3F"
mkdir -p "$OUT"

cd "$BASELINES_DIR/s3f"
PYTHONPATH="$BASELINES_DIR/s3f:$S3F_HOME${PYTHONPATH:+:$PYTHONPATH}" \
python "$BASELINES_DIR/s3f/compute_fitness.py" \
    -c "$S3F_CONFIG" \
    --ProteinGym_reference_file "$REF_CSV" \
    --output_scores_folder "$OUT" \
    --scoring_method $S3F_SCORING_METHODS \
    --model_name "$S3F_MODEL_NAME" \
    --device_ids "$GPU"
