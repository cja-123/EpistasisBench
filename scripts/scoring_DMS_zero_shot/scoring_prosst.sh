#!/usr/bin/env bash
# Zero-shot scoring with ProSST on the EpistasisBench assays.
#
# Scoring methods supported by epistasisbench/baselines/prosst/compute_fitness.py:
#   wildtype_marginal      wild-type marginal, one residue masked at a time
#   masked_marginal_a      all mutated positions masked simultaneously
#   joint_marginal         order-dependent joint marginal (Monte Carlo above depth 5)
#   persudo_loglikelihood  pseudo log-likelihood (spelling is kept as in the code)
#
# Several methods can be passed at once, e.g.
#   PROSST_SCORING_METHODS="wildtype_marginal joint_marginal" ./scoring_prosst.sh
#
# Required inputs, one file per assay named {DMS_id}.fasta / {DMS_id}.csv:
#   RESIDUE_DIR   wild-type residue fasta
#   STRUCTURE_DIR 3Di structure fasta built by foldseek from PDB_DIR
#   DMS_DIR       substitution csv (mutant column)
#
# Output: $OUT_DIR/prosst/{DMS_id}.csv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../zero_shot_config.sh"

PROSST_MODEL_NAME="${PROSST_MODEL_NAME:-ProSST-2048}"   # column prefix in the output csv
PROSST_MODEL_DIR="${PROSST_MODEL_DIR:?set PROSST_MODEL_DIR to the ProSST weights}"
PROSST_SCORING_METHODS="${PROSST_SCORING_METHODS:-wildtype_marginal}"
PROSST_DEPTH_THRESHOLD="${PROSST_DEPTH_THRESHOLD:-5}"

: "${RESIDUE_DIR:?set RESIDUE_DIR to the per-assay residue fasta directory}"
: "${STRUCTURE_DIR:?set STRUCTURE_DIR to the per-assay foldseek 3Di fasta directory}"

OUT="$OUT_DIR/ProSST-2048"
mkdir -p "$OUT"

python "$BASELINES_DIR/prosst/compute_fitness.py" \
    --model_name "$PROSST_MODEL_NAME" \
    --model_dir "$PROSST_MODEL_DIR" \
    --DMS_id_csv "$REF_CSV" \
    --residue_dir "$RESIDUE_DIR" \
    --structure_dir "$STRUCTURE_DIR" \
    --mutant_dir "$DMS_DIR" \
    --output_scores_folder "$OUT" \
    --scoring_method $PROSST_SCORING_METHODS \
    --depth_threshold "$PROSST_DEPTH_THRESHOLD" \
    --devices "$GPU" \
    --batch_size "$BATCH_SIZE" \
    --n_jobs "$N_JOBS"
