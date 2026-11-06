"""Compute EpistasisBench metrics for one or more models.

Two tasks are evaluated per assay and then averaged over assays:

* Task 1, zero-shot epistatic ddG prediction: Spearman rho between model scores and
  the measured ``DMS_score`` of double mutants, restricted to the ``sign`` and
  ``significant`` subsets, plus the same correlation on their single-mutant components.
* Task 2, zero-shot ddG_int scoring: the non-additive score
  ``S_int = score(AB) - score(A) - score(B)`` is compared with the measured ``ddG_int``.
  Reported metrics are Spearman rho, partial rho (conditioning on ``ddG_A + ddG_B``),
  precision, recall, F1, MCC, AUC, AUPRC and top 1 / 2.5 / 5 / 10 % recall.

Inputs
------
--label-csv  one row per double mutant, columns:
             uniprot_id, pdb_id, double_mutant, mutation_A, mutation_B,
             ddG_AB, ddG_A, ddG_B, ddG_int, thermodynamic_coupling_value,
             is_sign_epistasis, is_significant_epistasis
--score-dir  one csv per assay named ``{dms}.csv``, columns:
             mutant (``A123B`` or ``A123B:C456D``), DMS_score, one column per model.
             The assay id is ``{uniprot_id}_Tsuboyama_2023_{pdb_id}``.

Usage
-----
python compute_metrics.py --label-csv ../../data/EpistasisBench.csv \
    --score-dir ../../outputs/scores --output ../../outputs/metrics.csv \
    --models SaProt_650M_AF2 ESM3 S3F ProSST-2048
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TASK2_METRICS = [
    'spearman', 'partial_spearman', 'precision', 'recall', 'f1', 'mcc',
    'auc', 'auprc', 'top1_recall', 'top2.5_recall', 'top5_recall', 'top10_recall',
]


def calc_top_n_percent_recall(y_true, y_pred, percent):
    """Overlap between the true top-k and the predicted top-k, k = percent% of the samples."""
    n_samples = len(y_true)
    if n_samples == 0:
        return np.nan
    if np.nanstd(y_pred) == 0:
        return np.nan
    k = int(n_samples * percent / 100.0)
    k = min(max(k, 1), n_samples)
    true_top_k_indices = set(np.argsort(-y_true)[:k])
    pred_top_k_indices = set(np.argsort(-y_pred)[:k])
    return len(true_top_k_indices.intersection(pred_top_k_indices)) / k


def partial_spearman_corr(x, y, z):
    """Spearman correlation between x and y after removing the effect of the confounder z."""
    r_xy, _ = spearmanr(x, y)
    r_xz, _ = spearmanr(x, z)
    r_yz, _ = spearmanr(y, z)
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))


def load_labels(label_csv):
    labels = pd.read_csv(label_csv)
    labels['dms'] = labels.uniprot_id + '_Tsuboyama_2023_' + labels.pdb_id
    return labels


def compute_task1(labels, score_dir, models, min_mutants, tc_threshold):
    labels = labels.copy()
    labels['significant'] = labels.thermodynamic_coupling_value.abs() > tc_threshold

    rows = []
    for dms in sorted(set(labels.dms)):
        dt = pd.read_csv(Path(score_dir) / f'{dms}.csv')[['mutant', 'DMS_score'] + models]
        df = labels.loc[labels.dms == dms]
        subsets = {'sign': df.loc[df.is_sign_epistasis], 'significant': df.loc[df.significant]}

        for name, sub in subsets.items():
            if len(sub.index) < min_mutants:
                continue
            for model in models:
                tp = dt.loc[dt.mutant.isin(sub.double_mutant)].dropna(subset=[model])
                corr = spearmanr(tp.DMS_score, tp[model])[0] if len(tp) >= min_mutants else np.nan
                rows.append((dms, model, name, corr))

                tp = dt.loc[dt.mutant.isin(sub.mutation_A) | dt.mutant.isin(sub.mutation_B)].dropna(
                    subset=[model])
                corr = spearmanr(tp.DMS_score, tp[model])[0] if len(tp) >= min_mutants else np.nan
                rows.append((dms, model, f'{name}_single', corr))

    per_assay = pd.DataFrame(rows, columns=['dms', 'model', 'dataset', 'spearman'])

    summary = pd.DataFrame()
    for model in models:
        for dataset in ['sign', 'significant']:
            for suffix in ['', '_single']:
                values = per_assay.loc[
                    (per_assay.model == model) & (per_assay.dataset == dataset + suffix)].spearman
                summary.at[model, dataset + suffix] = values.mean()
                summary.at[model, dataset + suffix + '_sd'] = values.std(ddof=1)
    return per_assay, summary


def compute_task2(labels, score_dir, models, min_mutants):
    rows = []
    for dms in sorted(set(labels.dms)):
        dt = pd.read_csv(Path(score_dir) / f'{dms}.csv')
        tp_all = labels.loc[labels.dms == dms].copy()

        for model in models:
            mutant2score = dict(zip(dt.mutant, dt[model]))

            tp = tp_all.copy()
            tp['score'] = tp.double_mutant.map(mutant2score)
            tp['score_A'] = tp.mutation_A.map(mutant2score)
            tp['score_B'] = tp.mutation_B.map(mutant2score)

            tp = tp.dropna(subset=['score', 'score_A', 'score_B'])
            if len(tp) < min_mutants:
                rows.append((dms, model) + (np.nan,) * len(TASK2_METRICS))
                continue

            tp['S_int'] = tp.score - tp.score_A - tp.score_B
            y_true = tp.ddG_int.values
            y_pred = tp.S_int.values

            corr, _ = spearmanr(y_true, y_pred)
            partial_corr = partial_spearman_corr(y_true, y_pred, tp.ddG_A.values + tp.ddG_B.values)

            y_true_bin = (y_true > 0).astype(int)
            y_pred_bin = (y_pred > 0).astype(int)
            if len(np.unique(y_true_bin)) == 2:
                prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
                auc = roc_auc_score(y_true_bin, y_pred)
                auprc = average_precision_score(y_true_bin, y_pred)
            else:
                prec = rec = f1 = mcc = auc = auprc = np.nan

            rows.append((
                dms, model, corr, partial_corr, prec, rec, f1, mcc, auc, auprc,
                calc_top_n_percent_recall(y_true, y_pred, 1),
                calc_top_n_percent_recall(y_true, y_pred, 2.5),
                calc_top_n_percent_recall(y_true, y_pred, 5),
                calc_top_n_percent_recall(y_true, y_pred, 10),
            ))

    per_assay = pd.DataFrame(rows, columns=['dms', 'model'] + TASK2_METRICS)

    summary = pd.DataFrame()
    for model in models:
        tp = per_assay.loc[per_assay.model == model]
        for col in TASK2_METRICS:
            values = tp[col].dropna()
            summary.at[model, col] = values.mean() if len(values) else np.nan
            if 'spearman' in col and len(values):
                summary.at[model, f'{col}_sd'] = values.std(ddof=1)

    for col in ('partial_spearman_sd', 'spearman_sd'):
        if col not in summary.columns:
            summary[col] = np.nan
    summary = summary.loc[models][['partial_spearman_sd', 'spearman_sd'] + TASK2_METRICS]
    return per_assay, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compute EpistasisBench metrics (Task 1 and Task 2).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--label-csv', default='data/EpistasisBench.csv',
                        help='label table, one row per double mutant')
    parser.add_argument('--score-dir', default='outputs/scores',
                        help='directory with one {dms}.csv score table per assay')
    parser.add_argument('--output', default='outputs/metrics.csv',
                        help='aggregated metrics csv')
    parser.add_argument('--per-assay-output', default=None,
                        help='optional csv with per-assay metrics')
    parser.add_argument('--models', nargs='+', default=None,
                        help='models to evaluate; default: every column of the score tables '
                             'except mutant and DMS_score')
    parser.add_argument('--min-mutants', type=int, default=30,
                        help='skip a subset with fewer double mutants')
    parser.add_argument('--tc-threshold', type=float, default=0.5,
                        help='|thermodynamic coupling| above this is treated as significant')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    label_csv = Path(args.label_csv)
    score_dir = Path(args.score_dir)
    if not label_csv.exists():
        sys.exit(f'label csv not found: {label_csv}')
    if not score_dir.is_dir():
        sys.exit(f'score directory not found: {score_dir}')

    labels = load_labels(label_csv)

    models = args.models
    if models is None:
        first = sorted(score_dir.glob('*.csv'))[0]
        columns = pd.read_csv(first, nrows=0).columns
        models = [c for c in columns if c not in ('mutant', 'DMS_score')]

    sub_per_assay, sub_summary = compute_task1(
        labels, score_dir, models, args.min_mutants, args.tc_threshold)
    epi_per_assay, epi_summary = compute_task2(
        labels, score_dir, models, args.min_mutants)

    out = sub_summary.join(epi_summary, how='outer')
    out.index.name = 'model'
    out = out.map(lambda x: np.round(x, 3) if isinstance(x, float) else x)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output)

    if args.per_assay_output:
        Path(args.per_assay_output).parent.mkdir(parents=True, exist_ok=True)
        sub_per_assay.merge(epi_per_assay, on=['dms', 'model'], how='outer').to_csv(
            args.per_assay_output, index=False)

    print(f'models: {len(models)}')
    print(f'wrote {args.output}')
    return out


if __name__ == '__main__':
    main()
