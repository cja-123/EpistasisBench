#!/usr/bin/env bash
# Shared settings for the zero-shot scoring scripts in scoring_DMS_zero_shot/.
#
# Either edit the defaults below, or export the same variable names in your shell
# before calling a scoring script. Everything marked TODO has to point to your own
# copy of the model weights / structures; nothing here is machine specific.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="$REPO_ROOT/epistasisbench/baselines"

# ---------------------------------------------------------------- data ------
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"

# Reference table: one row per assay, needs DMS_id, target_seq, pdb_file, pdb_range.
# data/DMS_Tsuboyama.csv is released with the benchmark and lists the 50 assays used here.
REF_CSV="${REF_CSV:-$DATA_DIR/DMS_Tsuboyama.csv}"

# Per-assay substitution files: {DMS_id}.csv with columns mutant, mutated_sequence, DMS_score, DMS_score_bin.
DMS_DIR="${DMS_DIR:-$DATA_DIR/DMS_ProteinGym_substitutions}"

# TODO: directory with AlphaFold2-predicted structures ({pdb_file} as listed in REF_CSV).
# Used by ESM3 (--pdb_dir) and SaProt (--structure_data_folder, 3Di tokens are built from these).
PDB_DIR="${PDB_DIR:-}"

# TODO: ProSST inputs, one file per assay named {DMS_id}.fasta.
# RESIDUE_DIR  : wild-type residue sequence fasta
# STRUCTURE_DIR: 3Di structure sequence fasta, produced by foldseek from PDB_DIR
RESIDUE_DIR="${RESIDUE_DIR:-}"
STRUCTURE_DIR="${STRUCTURE_DIR:-}"

# Model identifiers as reported in the paper:
#   SaProt-650M-AF2 : westlake-repl/SaProt_650M_AF2, 3Di tokens built from AlphaFold2 structures
#   ESM3            : ESM3-open (1.4B), ESM3_MODEL_TYPE=esm3_open
#   ProSST-2048     : AI4Protein/ProSST-2048, structure vocabulary K=2048
#   S3F             : Zhang et al., NeurIPS 2024

OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs}"

# ------------------------------------------------------------- runtime ------
GPU="${GPU:-0}"                      # single device id, e.g. GPU=0 or GPU=1
BATCH_SIZE="${BATCH_SIZE:-1}"
N_JOBS="${N_JOBS:-1}"                # ProSST only

mkdir -p "$OUT_DIR"
