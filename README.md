# EpistasisBench

**EpistasisBench: Revealing the Limits of Zero-Shot Scoring Schemes for Protein Language Models**

EpistasisBench is a thermodynamically stratified zero-shot benchmark for protein language models. It defines two
tasks:

- **Task 1 — zero-shot epistatic ΔΔG prediction.** Spearman ρ between model scores and measured ΔΔG, evaluated on
  the isolated `sign` and `significant` epistasis subsets.
- **Task 2 — zero-shot ΔΔG_int scoring.** The model's non-additive score
  `S_int = score(AB) - score(A) - score(B)` is compared with the measured `ΔΔG_int` on all double mutants, without
  subset isolation, using Spearman ρ, partial ρ, precision / recall / F1 / MCC, AUC, AUPRC and top-K% recall.

## Benchmark at a glance

| | |
|---|---|
| Source data | Tsuboyama et al., 2023 (MegaScale), subset overlapping with ProteinGym |
| Data pool | 50,812 double mutations, 50 proteins, 180 residue pairs |
| `significant` epistasis | \|thermodynamic coupling\| > 0.5 kcal/mol — 6,003 double mutations, 35 proteins |
| `sign` epistasis | ε = 0.07 — 3,267 double mutations, 31 proteins |
| Evaluated models | 22 baselines in the paper (17 PLMs + 5 statistical / latent models); their scoring code is available in ProteinGym. This repository provides the 4 MLM baselines — SaProt-650M-AF2, ESM3, S3F, ProSST-2048 — with the new scoring schemes: Joint Marginal, Masked Marginal and Pseudo-log-likelihood (PLL) |

## Repository layout

```
EpistasisBench/
├── data/
│   ├── dataset_construction/
│   │   ├── dataset_construction.ipynb   # rebuilds the dataset from the files below
│   │   └── TC.py                        # thermodynamic coupling (MCMC additive model)
│   ├── DMS_substitutions.csv            # ProteinGym assay index
│   ├── DMS_ProteinGym_substitutions/    # 50 per-assay substitution csvs
│   ├── DMS_Tsuboyama.csv                # assay table used by the scoring scripts
│   ├── thermodynamic_coupling.csv       # 6 columns, needed only for a rebuild (22 MB)
│   ├── epi.csv                          # intermediate of a rebuild
│   └── EpistasisBench.csv               # 50,812 x 12 released label table
├── epistasisbench/
│   ├── baselines/
│   │   ├── esm3/    compute_fitness.py, esm/            # ESM3 / ESM-C
│   │   ├── saprot/  compute_fitness.py, foldseek_util.py
│   │   ├── prosst/  compute_fitness.py
│   │   └── s3f/     compute_fitness.py, util.py, s3f/, script/, config/, s3f_environment.yml
│   └── evaluation/
│       ├── compute_metrics.py           # CLI
│       └── compute_metrics.ipynb        # same metrics, step by step
├── environment/
│   ├── proteingym_env.txt               # SaProt environment (pip freeze)
│   ├── esm3_environment.yml
│   ├── prosst_environment.yaml
│   └── eval_environment.yml             # metrics only (pandas >= 2.1)
└── scripts/
    ├── zero_shot_config.sh              # shared paths and runtime settings
    └── scoring_DMS_zero_shot/
        ├── scoring_esm3.sh
        ├── scoring_saprot.sh
        ├── scoring_prosst.sh
        └── scoring_s3f.sh
```

## Installation

Each baseline runs in its own environment; evaluation uses a separate, lighter one.

```bash
# SaProt
conda create -n saprot python=3.10 -y && conda activate saprot
pip install -r environment/proteingym_env.txt

# ESM3 / ProSST / S3F
conda env create -f environment/esm3_environment.yml
conda env create -f environment/prosst_environment.yaml
conda env create -f epistasisbench/baselines/s3f/s3f_environment.yml

# evaluation (pandas >= 2.1)
conda env create -f environment/eval_environment.yml
```

Do not install `proteingym_env.txt` into the ESM3 / ProSST / S3F environments: it pins
`torch==1.13.1` and will downgrade the model environments.

Model weights are not shipped: point the scripts at your local checkpoints (see below) or let the model download
from HuggingFace.

## Running the baselines

Every scoring script writes one csv per assay to `outputs/<MODEL>/`; SaProt writes to
`outputs/SaProt_650M_AF2/{scoring_method}/{DMS_id}.csv`, one directory per scoring method. Inputs are the shipped
`data/DMS_Tsuboyama.csv` (assay table) and `data/DMS_ProteinGym_substitutions/` (per-assay substitution csv);
override them with `REF_CSV` / `DMS_DIR` if needed.

```bash
export PDB_DIR=/path/to/alphafold2_structures          # used by ESM3 and SaProt
export SAPROT_MODEL_PATH=/path/to/SaProt_650M_AF2
./scripts/scoring_DMS_zero_shot/scoring_saprot.sh

export ESM3_MODEL_TYPE=esm3_open                       # esm3_open | esmc_300M | esmc_600M
export ESM3_MODEL_PATH=/path/to/esm3_open              # optional, empty means download
./scripts/scoring_DMS_zero_shot/scoring_esm3.sh

export PROSST_MODEL_DIR=/path/to/ProSST-2048
export RESIDUE_DIR=/path/to/residue_sequence           # {DMS_id}.fasta
export STRUCTURE_DIR=/path/to/structure_sequence       # foldseek 3Di fasta, {DMS_id}.fasta
./scripts/scoring_DMS_zero_shot/scoring_prosst.sh

export S3F_CONFIG=/path/to/s3f.yaml
./scripts/scoring_DMS_zero_shot/scoring_s3f.sh
```

Shared settings live in `scripts/zero_shot_config.sh` and can all be overridden by environment variables:
`REF_CSV`, `DMS_DIR`, `PDB_DIR`, `RESIDUE_DIR`, `STRUCTURE_DIR`, `OUT_DIR`, `GPU`, `BATCH_SIZE`, `N_JOBS`.

The other 18 of the 22 baselines reuse ProteinGym's scoring code
(https://github.com/OATML-Markslab/ProteinGym); only the four MLMs here add the new scoring schemes. All four
support Joint Marginal and PLL, and Masked Marginal is exposed as `default` for ESM3 / SaProt and as
`masked_marginal` / `masked_marginal_a` for ESM3 / ProSST / S3F. Running one model with the three methods
reproduces the scoring-scheme comparison of the paper.

Scoring methods (`--scoring_method`):

| model | methods |
|---|---|
| ESM3 | `default`, `joint_marginal`, `mask_marginal`, `pseudo_loglikelihood` |
| SaProt | `default`, `joint_marginal`, `pll`, `nopll` |
| ProSST | `wildtype_marginal`, `joint_marginal`, `masked_marginal_a`, `persudo_loglikelihood` |
| S3F | `masked_marginal_a`, `joint_marginal`, `pseudo_loglikelihood` |

## Baseline sources

| model | code | paper |
|---|---|---|
| SaProt-650M-AF2 | https://github.com/westlake-repl/SaProt | https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5 |
| ESM3 | https://github.com/evolutionaryscale/esm | https://www.science.org/doi/10.1126/science.ads0018 |
| S3F | https://github.com/DeepGraphLearning/S3F | https://papers.nips.cc/paper_files/paper/2024/hash/b7d795e655c1463d7299688d489e8ef4-Abstract-Conference.html |
| ProSST-2048 | https://github.com/ai4protein/ProSST | https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ed57b293db0aab7cc30c44f45262348-Abstract-Conference.html |
| other 18 baselines (22 in total) | https://github.com/OATML-Markslab/ProteinGym | see ProteinGym |

## Computing metrics

Run these in the `epibench-eval` environment (`environment/eval_environment.yml`); the script needs
pandas >= 2.1.

Merge the per-model score files into one csv per assay in `outputs/scores/{DMS_id}.csv` (columns `mutant`,
`DMS_score`, one column per model), then:

```bash
python epistasisbench/evaluation/compute_metrics.py \
    --label-csv data/EpistasisBench.csv \
    --score-dir outputs/scores \
    --output outputs/metrics.csv \
    --models SaProt_650M_AF2 ESM3 S3F ProSST-2048
```

`--per-assay-output per_assay.csv` additionally writes the per-assay values. The same computation is available
cell by cell in `epistasisbench/evaluation/compute_metrics.ipynb`.

Task 1 keeps only assays with at least 30 double mutants in the subset (`--min-mutants`); Task 2 uses every
double mutant of all assays, with no subset filter. `partial_spearman` conditions on `ddG_A + ddG_B`, and AUC
treats stabilizing interactions (`ddG_int > 0`) as the positive class.

## Rebuilding the dataset (optional)

`data/` already ships everything the baselines and the metrics need. Rebuild only if you want to reproduce
`data/EpistasisBench.csv` from the original sources:

| file | source |
|---|---|
| `DMS_substitutions.csv` | ProteinGym, `reference_files/DMS_substitutions.csv` (https://github.com/OATML-Markslab/ProteinGym) |
| `DMS_ProteinGym_substitutions/{DMS_id}.csv` | ProteinGym per-assay substitution files |
| `thermodynamic_coupling.csv` | output of `data/dataset_construction/TC.py`, computed from the Tsuboyama 2023 MegaScale Dataset2 + Dataset3 (https://zenodo.org/record/7992926). Shipped here with the 6 columns the notebook reads; `TC.py` writes the full table |

Then open `data/dataset_construction/dataset_construction.ipynb` and run all cells. It reads the three files above
and writes `data/DMS_Tsuboyama.csv`, `data/epi.csv` and `data/EpistasisBench.csv`.

## Citation

```bibtex
@misc{chen2026epistasisbench,
  title  = {EpistasisBench: Revealing the Limits of Zero-Shot Scoring Schemes for Protein Language Models},
  author = {Chen, Junan and Min, Jianwen and He, Liang and Zhu, Yiheng},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

Released under the MIT License (see `LICENSE`). Third-party model weights
(e.g., ESM3, ProSST) are not included and remain under their own licenses.

The `epistasisbench/baselines/esm3/esm/` directory is a copy of the upstream
ESM package (https://github.com/evolutionaryscale/esm) and is distributed under
the Cambrian Open License Agreement, not MIT; see
`epistasisbench/baselines/esm3/LICENSE-ESM3`. The `s3f/s3f/` directory follows
the licence of the upstream S3F repository.
