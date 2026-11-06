import torch
import os
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader
from itertools import product, permutations
from joblib import Parallel, delayed

def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)

def parse_mutant(mutant):
    mutations = []
    for sub_mutant in mutant.split(":"):
        wt = sub_mutant[0]
        idx = int(sub_mutant[1:-1]) - 1
        mt = sub_mutant[-1]
        mutations.append((wt, idx, mt))
    return mutations

def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor(
        [ shift_structure_sequence, ],
        dtype=torch.long,
    )

@torch.no_grad()
def wildtype_marginal(model, tokenizer, residue_fasta, structure_fasta, mutant_df, batch_size=None, device='cuda', n_jobs=None, depth_threshold=None):
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)
    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(device)
    attention_mask = tokenized_results["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        labels=input_ids,
    )
    logits = outputs.logits
    logits = torch.log_softmax(logits[:, 1:-1, :], dim=-1)
    mutants = mutant_df["mutant"].tolist()
    scores = []
    vocab = tokenizer.get_vocab()
    for mutant in tqdm(mutants):
        pred_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            score = logits[0, idx, vocab[mt]] - logits[0, idx, vocab[wt]]
            pred_score += score.item()
        scores.append(pred_score)
    return scores

@torch.no_grad()
def masked_marginal_a(model, tokenizer, residue_fasta, structure_fasta, mutant_df, batch_size=None, device='cuda', n_jobs=32, depth_threshold=None):
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)
    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    
    mutants = mutant_df["mutant"].tolist()
    vocab = tokenizer.get_vocab()

    in_seq = []
    mutations_list = []
    
    for mutant in mutants:
        mutations = parse_mutant(mutant)
        mutations_list.append(mutations)
        
        pred_score = 0
        
        mask_seq = list(sequence)
        for _, idx, _ in mutations:
            mask_seq[idx] = '<mask>'
        mask_seq = ''.join(mask_seq)
        in_seq.append(mask_seq)
        
    # tokenized = tokenizer(in_seq, return_tensors="pt")
    # dataset = TensorDataset(
    #     tokenized["input_ids"], 
    #     tokenized["attention_mask"],
    # )
    results = Parallel(n_jobs=n_jobs)(
        delayed(tokenizer)([i], return_tensors="pt") for i in in_seq
    )
    input_ids, attention_mask = [], []
    for res in results:
        input_ids.append(res["input_ids"])
        attention_mask.append(res["attention_mask"])
    input_ids = torch.concat(input_ids, dim=0)
    attention_mask = torch.concat(attention_mask, dim=0)
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batch_logits = []
    for batch in tqdm(dataloader, ascii=True, desc='forward'):
        input_ids, attn_mask = [x.to(device) for x in batch]
        
        ss_batch = ss_input_ids.expand(len(input_ids), -1)
        
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attn_mask, 
            ss_input_ids=ss_batch, 
            labels=input_ids
        )
        logits = outputs.logits[:, 1:-1, :]
        batch_logits.append(logits.detach().cpu())

    # logits shape: (total_mutants, seq_len, vocab_size)
    logits = torch.concat(batch_logits, dim=0)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    scores = []
    for i, mutations in enumerate(mutations_list):
        score = 0
        for wt, idx, mt in mutations:
            score += log_probs[i, idx, vocab[mt]] - log_probs[i, idx, vocab[wt]]
        scores.append(score.item())
    
    return scores


def get_code2combo(depth):
    print(depth)
    code2combo = dict()
    for i in product([0, 1], repeat=depth):
        if 1 in i:
            code, combo = [], []
            for j, bit in enumerate(i):
                if bit == 1:
                    code.append(str(j))
                    combo.append(j)
            code2combo[','.join(code)] = combo
    return code2combo


@torch.no_grad()
def joint_marginal(model, tokenizer, residue_fasta, structure_fasta, mutant_df, batch_size, device='cuda', n_jobs=32, depth_threshold = 5, mean_method='Geometric', mc_seed=42):
    repeat_times = 2 ** depth_threshold
    
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)
    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    vocab = tokenizer.get_vocab()
    
    depths = mutant_df["mutant"].apply(lambda x: len(x.split(':')))
    
    idx_eq_1 = depths[depths == 1].index
    idx_le_threshold = depths[(depths <= depth_threshold) & (depths > 1)].index
    idx_gt_threshold = depths[depths > depth_threshold].index

    scores_map = {}

    if not idx_eq_1.empty:
        deep_scores = masked_marginal_a(model, tokenizer, residue_fasta, structure_fasta, mutant_df.loc[idx_eq_1], batch_size, device=device)
        for idx, score in zip(idx_eq_1, deep_scores):
            scores_map[idx] = score

    if not idx_le_threshold.empty:
        df_joint = mutant_df.loc[idx_le_threshold]
        
        unique_depths = sorted(depths[idx_le_threshold].unique())

        for depth in unique_depths:
            current_indices = depths[idx_le_threshold][depths == depth].index
            tb = mutant_df.loc[current_indices]

            # code2combo
            # {'2': [2], '1': [1], '1,2': [1, 2], '0': [0], '0,2': [0, 2], '0,1': [0, 1], '0,1,2': [0, 1, 2]}
            code2combo = get_code2combo(depth)
            
            in_seq, mutant_code, mutant2muts = [], [], dict()
            for mutant in tqdm(tb.mutant, ascii=True, desc='make input'):
                muts = []
                sequence_wt, sequence_mt = list(sequence), list(sequence)
                for mut in mutant.split(":"):
                    wt, pos, mt = mut[0], int(mut[1:-1]) - 1, mut[-1]
                    muts.append((wt, pos, mt))
                    sequence_mt[pos] = mt
                    assert sequence_wt[pos] == wt

                mutant2muts[mutant] = muts
                
                for code in code2combo:
                    # combo is idx to be masked
                    combo = code2combo[code]

                    # -- mask seq --
                    seq_wt, seq_mt = sequence_wt + [], sequence_mt + []
                    for idx in combo:
                        seq_wt[muts[idx][1]] = '<mask>'
                        seq_mt[muts[idx][1]] = '<mask>'

                    seq_wt, seq_mt = ''.join(seq_wt), ''.join(seq_mt)
                    
                    # wt
                    mutant_code.append(f'{mutant}|{code}|wt')
                    in_seq.append(seq_wt)

                    # mt
                    mutant_code.append(f'{mutant}|{code}|mt')
                    in_seq.append(seq_mt)
                        
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(tokenizer)([i], return_tensors="pt") for i in in_seq
            )
            input_ids, attention_mask = [], []
            for res in results:
                input_ids.append(res["input_ids"])
                attention_mask.append(res["attention_mask"])
            input_ids = torch.concat(input_ids, dim=0)
            attention_mask = torch.concat(attention_mask, dim=0)
            dataset = TensorDataset(input_ids, attention_mask)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            batch_logits = []
            for batch in tqdm(dataloader, ascii=True, desc='predicting batch'):
                input_ids, attn_mask = [x.to(device) for x in batch]
                ss_batch = ss_input_ids.expand(len(input_ids), -1) # ?
                
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, ss_input_ids=ss_batch, labels=input_ids)
                logits = outputs.logits[:, 1:-1, :]
                batch_logits.append(logits.detach().cpu())

            logits = torch.concat(batch_logits, dim=0)
            
            probs = torch.softmax(logits, dim=-1)
            code2probs_tensor = dict(zip(mutant_code, list(probs)))

            all_idx = set(range(depth))
            orders = list(permutations(range(depth))) # [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            joint_scores = []
            for mutant in tqdm(tb.mutant, ascii=True, desc='make output'):
                muts = mutant2muts[mutant]

                code2prob = dict()
                for code in code2combo:
                    combo = code2combo[code]

                    probs_tensor_wt = code2probs_tensor[f'{mutant}|{code}|wt']
                    probs_tensor_mt = code2probs_tensor[f'{mutant}|{code}|mt']

                    for idx in combo:
                        wt, pos, mt = muts[idx]
                        code2prob[f'wt|{code}|{idx}'] = probs_tensor_wt[pos, vocab[wt]].item()
                        code2prob[f'mt|{code}|{idx}'] = probs_tensor_mt[pos, vocab[mt]].item()

                if mean_method == 'Arithmetic':
                    P_wt, P_mt = [], []
                    for order in orders:
                        P_wt_order, P_mt_order = 1, 1
                        for i in range(depth):
                            # (0, 2, 1) = 0,1,2/0 * 1,2/2 * 1/1
                            code = ','.join([str(idx) for idx in sorted(all_idx - set(order[:i]))])
                            idx = order[i]
                            P_wt_order *= code2prob[f'wt|{code}|{idx}']
                            P_mt_order *= code2prob[f'mt|{code}|{idx}']
                        P_wt.append(P_wt_order)
                        P_mt.append(P_mt_order)
    
                    score = np.log(np.sum(P_mt) + 1e-30) - np.log(np.sum(P_wt) + 1e-30)
                elif mean_method == 'Geometric':
                    score = []
                    for order in orders:
                        score_order = 0
                        for i in range(depth):
                            # (0, 2, 1) = 0,1,2/0 * 1,2/2 * 1/1
                            code = ','.join([str(idx) for idx in sorted(all_idx - set(order[:i]))])
                            idx = order[i]
                    
                            P_wt = code2prob[f'wt|{code}|{idx}']
                            P_mt = code2prob[f'mt|{code}|{idx}']
                    
                            score_order += np.log(P_mt + 1e-30) - np.log(P_wt + 1e-30)
                        score.append(score_order)
                    
                    score = np.sum(score) / len(score)
                else:
                    raise # mean_method not in ('Arithmetic', 'Geometric')
                    
                joint_scores.append(score)
            
            for original_idx, score in zip(current_indices, joint_scores):
                scores_map[original_idx] = score

    if not idx_gt_threshold.empty:
        df_joint = mutant_df.loc[idx_gt_threshold]
        
        unique_depths = sorted(depths[idx_gt_threshold].unique())

        for depth in unique_depths:
            current_indices = depths[idx_gt_threshold][depths == depth].index
            tb = mutant_df.loc[current_indices]

            np.random.seed(mc_seed)
            num_mutants = len(tb.mutant)
            seeds = np.random.randint(low=0, high=1048576, size=(num_mutants, repeat_times), dtype=int)
            
            # for dataloader input
            def _func(mutant, seed_list):
                in_seq, mutant_code = [], []
                
                # make sequence_mt and sequence_wt
                muts = []
                sequence_wt, sequence_mt = list(sequence), list(sequence)
                for mut in mutant.split(":"):
                    wt, pos, mt = mut[0], int(mut[1:-1]) - 1, mut[-1]
                    muts.append((wt, pos, mt))
                    sequence_mt[pos] = mt
                    assert sequence_wt[pos] == wt

                code_combo = []
                for repeat_idx in range(repeat_times):
                    seed = seed_list[repeat_idx]
                    
                    np.random.seed(seed)
                    size = np.random.randint(low=1, high=depth + 1)
                    combo = sorted(np.random.choice(depth, size, replace=False).tolist())
                    code = ','.join(map(str, combo))
                    
                    code_combo.append((code, combo)) # useful later
                    
                    # -- mask seq --
                    seq_wt, seq_mt = sequence_wt + [], sequence_mt + []
                    for idx in combo:
                        seq_wt[muts[idx][1]] = '<mask>'
                        seq_mt[muts[idx][1]] = '<mask>'

                    seq_wt, seq_mt = ''.join(seq_wt), ''.join(seq_mt)

                    
                    wt_code = f'{mutant}|{code}|wt'
                    mt_code = f'{mutant}|{code}|mt'
                    if wt_code not in mutant_code and mt_code not in mutant_code:
                        # wt
                        mutant_code.append(wt_code)
                        in_seq.append(seq_wt)
    
                        # mt
                        mutant_code.append(mt_code)
                        in_seq.append(seq_mt)
                    else:
                        assert wt_code in mutant_code and mt_code in mutant_code

                return in_seq, mutant_code, muts, code_combo

            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_func)(mutant, seeds[mutant_idx]) for mutant_idx, mutant in enumerate(tb.mutant)
            )
            in_seq, mutant_code, muts, code_combo = list(zip(*results))

            in_seq = sum(in_seq, [])
            mutant_code = sum(mutant_code, [])
            mutant2muts = dict(zip(tb.mutant, muts))
            mutant2code_combo = dict(zip(tb.mutant, code_combo))

            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(tokenizer)([i], return_tensors="pt") for i in in_seq
            )
            input_ids, attention_mask = [], []
            for res in results:
                input_ids.append(res["input_ids"])
                attention_mask.append(res["attention_mask"])
            input_ids = torch.concat(input_ids, dim=0)
            attention_mask = torch.concat(attention_mask, dim=0)
            dataset = TensorDataset(input_ids, attention_mask)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            batch_logits = []
            for batch in tqdm(dataloader, ascii=True, desc='mc predicting batch'):
                input_ids, attn_mask = [x.to(device) for x in batch]
                ss_batch = ss_input_ids.expand(len(input_ids), -1) # ?
                
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, ss_input_ids=ss_batch, labels=input_ids)
                logits = outputs.logits[:, 1:-1, :]
                batch_logits.append(logits.detach().cpu())

            logits = torch.concat(batch_logits, dim=0)
            
            probs = torch.softmax(logits, dim=-1)
            code2probs_tensor = dict(zip(mutant_code, list(probs)))

            joint_scores = []
            for mutant in tqdm(tb.mutant, ascii=True, desc='make mc output'):
                
                muts = mutant2muts[mutant]
                code_combo_list = mutant2code_combo[mutant]
                
                score = 0
                for code, combo in code_combo_list:
                    # combo is <mask> idx
                    size = len(combo)
                    
                    probs_tensor_wt = code2probs_tensor[f'{mutant}|{code}|wt']
                    probs_tensor_mt = code2probs_tensor[f'{mutant}|{code}|mt']

                    combo_score = 0
                    for idx in combo:
                        wt, pos, mt = muts[idx]
                        
                        P_wt_idx = probs_tensor_wt[pos, vocab[wt]].item()
                        P_mt_idx = probs_tensor_mt[pos, vocab[mt]].item()
                        
                        combo_score += np.log(P_mt_idx + 1e-30) - np.log(P_wt_idx + 1e-30)
                    score += combo_score * depth / size
                score /= repeat_times
                    
                joint_scores.append(score)
            
            for original_idx, score in zip(current_indices, joint_scores):
                scores_map[original_idx] = score
                
    return [scores_map[i] for i in mutant_df.index]


def read_names(DMS_id_csv):
    return list(pd.read_csv(DMS_id_csv).DMS_id)

def dynamic_batch_size(func, batch_size):
    print(f'>>> batch_size =  {batch_size}', flush=True)
    while batch_size > 0:
        try:
            return func(batch_size=batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size = batch_size // 2
                print(f'>>> OOM, now trying with batch_size = {batch_size}', flush=True)
            else:
                raise # maybe other err


# -----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def _compute_pll_score(sequence, model, tokenizer, ss_input_ids, device, n_jobs, batch_size):
    seq_len = len(sequence)
    
    masked_seqs = []
    for i in range(seq_len):
        temp_seq = list(sequence)
        temp_seq[i] = '<mask>'
        masked_seqs.append("".join(temp_seq))
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(tokenizer)([s], return_tensors="pt") for s in masked_seqs
    )
    input_ids = torch.cat([r["input_ids"] for r in results], dim=0)
    attention_mask = torch.cat([r["attention_mask"] for r in results], dim=0)
    
    # shape: (seq_len,)
    target_ids = tokenizer(sequence, return_tensors="pt")["input_ids"][0, 1:-1].to(device)
    
    dataset = TensorDataset(input_ids, attention_mask)
    current_batch_size = batch_size if batch_size else 32
    dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=False)
    
    total_log_prob = 0.0
    current_idx = 0
    
    for batch in dataloader:
        batch_input_ids, batch_attn_mask = [x.to(device) for x in batch]
        b_size = len(batch_input_ids)
        
        ss_batch = ss_input_ids.expand(b_size, -1)
        
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attn_mask,
            ss_input_ids=ss_batch,
            labels=batch_input_ids
        )
        
        logits = outputs.logits[:, 1:-1, :] 
        log_probs = torch.log_softmax(logits, dim=-1)
        
        rows = torch.arange(b_size, device=device)
        cols = torch.arange(current_idx, current_idx + b_size, device=device)
        tokens = target_ids[cols]
        
        batch_probs = log_probs[rows, cols, tokens]
        total_log_prob += batch_probs.sum().item()
        
        current_idx += b_size
        
    return total_log_prob

@torch.no_grad()
def persudo_loglikelihood(model, tokenizer, residue_fasta, structure_fasta, mutant_df, batch_size=None, device='cuda', n_jobs=1, depth_threshold=None):
    wt_sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)
    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    
    mutants = mutant_df["mutant"].tolist()
    
    print("Computing Wildtype Pseudo Log Likelihood...")
    wt_pll = _compute_pll_score(wt_sequence, model, tokenizer, ss_input_ids, device, n_jobs, batch_size)
    print(f"Wildtype PLL: {wt_pll}")
    
    scores = []
    
    for mutant in tqdm(mutants, ascii=True):
        seq_list = list(wt_sequence)
        mutations = parse_mutant(mutant)
        for wt, idx, mt in mutations:
            seq_list[idx] = mt
        mt_sequence = "".join(seq_list)
        
        mt_pll = _compute_pll_score(mt_sequence, model, tokenizer, ss_input_ids, device, n_jobs, batch_size)
        
        scores.append(mt_pll - wt_pll)
        
    return scores

# -----------------------------------------------------------------------------------------------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="AI4Protein/ProSST-2048", required=True)
    parser.add_argument("--model_dir", type=str, default=None, required=True)
    # parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--DMS_id_csv", type=str, default=None, required=True)
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory containing all data",)
    parser.add_argument("--residue_dir", type=str, default=None, help="Directory containing FASTA files of residue sequences",)
    parser.add_argument("--structure_dir", type=str, default=None, help="Directory containing FASTA files of structure sequences",)
    parser.add_argument("--mutant_dir", type=str, default=None, help="Directory containing CSV files with mutants",)
    parser.add_argument("--output_scores_folder", default=None, help="Directory to save scores")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mutant_chunk_size", type=int, default=50000)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--devices", type=int, default=0, nargs="+")
    parser.add_argument("--scoring_method", type=str, default=None, nargs="+", required=True)
    parser.add_argument("--depth_threshold", type=int, default=5)
    parser.add_argument("--no_overwrite", action="store_true", help="if true, won't overwrite")
    args = parser.parse_args()

    device = torch.device(args.devices[0])
    device_ids = args.devices

    print("Scoring proteins...")
    os.makedirs(args.output_scores_folder, exist_ok=True)

    protein_names = read_names(args.DMS_id_csv)
    protein_names = sorted(protein_names)
    print(protein_names)
    print(f">>> total proteins: {len(protein_names)}")

    score_protein_dict = {
        'wildtype_marginal': wildtype_marginal,
        'joint_marginal': joint_marginal,
        'masked_marginal_a': masked_marginal_a,
        'persudo_loglikelihood': persudo_loglikelihood,
    }
    
    model_name, model_dir = args.model_name, args.model_dir
    print(f">>> Loading model {model_name} from {model_dir} ...")
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    # if args.lora_dir is not None:
    #     print(f'>>> Loading lora {args.lora_dir} ...')
    #     model = PeftModel.from_pretrained(model, args.lora_dir)
        
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device)
    
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    for scoring_method in args.scoring_method:
        print(f">>> Using {scoring_method}...")
        score_protein = score_protein_dict[scoring_method]

        for idx, protein_name in enumerate(protein_names):
            print(f">>> Scoring {protein_name}, current {idx+1}/{len(protein_names)}...")
            
            if args.base_dir:
                residue_fasta = f"{args.base_dir}/residue_sequence/{protein_name}.fasta"
                structure_fasta = f"{args.base_dir}/structure_sequence/{model_name.split('-')[-1]}/{protein_name}.fasta"
                mutant_file = f"{args.base_dir}/substitutions/{protein_name}.csv"
            
            if args.residue_dir:
                residue_fasta = f"{args.residue_dir}/{protein_name}.fasta"
            if args.structure_dir:
                structure_fasta = f"{args.structure_dir}/{protein_name}.fasta"
            if args.mutant_dir:
                mutant_file = f"{args.mutant_dir}/{protein_name}.csv"

            
            # remove the path from model_name if it contains path separators
            # save_model_name = model_name.split("/")[-1]
            save_model_name = model_name
            col_name = f'{save_model_name}.{scoring_method}'
            
            output_file = f"{args.output_scores_folder}/{protein_name}.csv"
            if os.path.exists(output_file):
                mutant_file = output_file

                _tb = pd.read_csv(mutant_file)
                if args.no_overwrite and col_name in _tb.columns:
                    print(f'>>> skip {protein_name} ...')
                    continue

            mutant_df = pd.read_csv(mutant_file)
            
            print(f'n_mutant: {len(mutant_df.index)}')
            scores = []
            chunk_size = args.mutant_chunk_size
            for i in range(0, len(mutant_df.index), chunk_size):
                chunk_df = mutant_df.iloc[i : i + chunk_size]
                scores.extend(
                    dynamic_batch_size(
                        lambda batch_size: score_protein(
                            model=model,
                            tokenizer=tokenizer,
                            residue_fasta=residue_fasta,
                            structure_fasta=structure_fasta,
                            mutant_df=chunk_df,
                            batch_size=batch_size,
                            device=device,
                            n_jobs=args.n_jobs,
                            depth_threshold=args.depth_threshold
                        ),
                        args.batch_size + 0
                    )
                )
            
            
            mutant_df[col_name] = scores
            
            corr = spearmanr(mutant_df["DMS_score"], mutant_df[col_name]).correlation
            print(f">>> {col_name} on {protein_name}: {corr}")
            mutant_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
