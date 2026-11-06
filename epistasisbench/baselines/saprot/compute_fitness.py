import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from foldseek_util import get_struc_seq
from itertools import product, permutations
from scipy.stats import spearmanr
from tqdm import tqdm

foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def mask_seq_saprot(seq_str, positions):
    assert len(seq_str) % 2 == 0
    seq_list = list(seq_str)
    for pos in positions:
        seq_list[pos * 2] = "#"
    return "".join(seq_list)

def mutate_seq_saprot(seq_str, positions, mt_aas):
    assert len(seq_str) % 2 == 0
    seq_list = list(seq_str)
    for pos, mt_aa in zip(positions, mt_aas):
        seq_list[pos * 2] = mt_aa
    return "".join(seq_list)

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def score_mutations_saprot(seq, mutations, model, tokenizer, scoring_method='default'):
    parsed_mutations = []
    for mutation in mutations:
        sub_mutations = mutation.split(":")
        multi_wt, multi_mt, multi_pos, multi_seq_pos = "", "", [], []
        
        for sub_mutation in sub_mutations:
            wt, pos_str, mt = sub_mutation[0], sub_mutation[1:-1], sub_mutation[-1]
            pos = int(pos_str)
            seq_pos = pos - 1
            if seq_pos < 0 or seq_pos >= len(seq): 
                print(seq)
                print(sub_mutation)
                raise
                break
            if seq[seq_pos * 2] != wt: raise ValueError("WT mismatch")
            multi_wt += wt; multi_mt += mt
            multi_pos.append(pos); multi_seq_pos.append(seq_pos)
        
        parsed_mutations.append((multi_wt, multi_pos, multi_mt, multi_seq_pos, mutation))

    if scoring_method == 'pll':
        return _score_mutations_pll_saprot(seq, parsed_mutations, model, tokenizer)
    elif scoring_method == 'joint_marginal':
        return _score_mutations_joint_saprot(seq, parsed_mutations, model, tokenizer)
    else:
        return _score_mutations_default_saprot(seq, parsed_mutations, model, tokenizer)

def _score_mutations_default_saprot(seq, parsed_mutations, model, tokenizer):
    mutation_scores = {}
    for wt, pos_list, mt, seq_pos_list, mutation_name in parsed_mutations:
        tokens = tokenizer.tokenize(seq)
        for seq_pos in seq_pos_list:
            tokens[seq_pos] = "#" + tokens[seq_pos][-1]
        mask_seq = " ".join(tokens)
        inputs = tokenizer(mask_seq, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            
        score = 0
        for i in range(len(seq_pos_list)):
            ori_aa, mut_aa, seq_pos = wt[i], mt[i], seq_pos_list[i]
            ori_st = tokenizer.get_vocab()[ori_aa + foldseek_struc_vocab[0]]
            mut_st = tokenizer.get_vocab()[mut_aa + foldseek_struc_vocab[0]]
            ori_prob = probs[0, seq_pos + 1, ori_st: ori_st + len(foldseek_struc_vocab)].sum()
            mut_prob = probs[0, seq_pos + 1, mut_st: mut_st + len(foldseek_struc_vocab)].sum()
            score += torch.log(mut_prob / ori_prob).item()
            
        mutation_scores[mutation_name] = score
    return mutation_scores

def _score_mutations_joint_saprot(seq, parsed_mutations, model, tokenizer, depth_threshold=5):
    if not parsed_mutations:
        return {}

    depths = [len(seq_pos_list) for _, _, _, seq_pos_list, _ in parsed_mutations]

    mma_indices = [i for i, depth in enumerate(depths) if depth == 1 or depth > depth_threshold]
    jm_indices = [i for i, depth in enumerate(depths) if 2 <= depth <= depth_threshold]
    
    # mma_indices = list(range(len(depths)))
    # jm_indices = []
    
    mutation_scores = {}

    if mma_indices:
        for idx in tqdm(mma_indices, ascii=True, desc='mma'):
            wt, pos_list, mt, seq_pos_list, mutation_name = parsed_mutations[idx]
            # print(parsed_mutations[idx])
            masked_seq = mask_seq_saprot(seq, seq_pos_list)
            # print(masked_seq)
            tokens = tokenizer.tokenize(masked_seq)
            # print(tokens)
            inputs = tokenizer(" ".join(tokens), return_tensors="pt").to(model.device)
            # print(inputs)
            
            with torch.no_grad():
                logits = model(**inputs).logits
                token_probs = torch.log_softmax(logits[0], dim=-1)

            # print(token_probs)
            score = 0.0
            for i, seq_pos in enumerate(seq_pos_list):
                ori_st = tokenizer.get_vocab()[wt[i] + foldseek_struc_vocab[0]]
                mut_st = tokenizer.get_vocab()[mt[i] + foldseek_struc_vocab[0]]
                wt_prob = token_probs[seq_pos + 1, ori_st: ori_st + len(foldseek_struc_vocab)].exp().sum()
                mt_prob = token_probs[seq_pos + 1, mut_st: mut_st + len(foldseek_struc_vocab)].exp().sum()
                score += torch.log(mt_prob / (wt_prob + 1e-30)).item()
            mutation_scores[mutation_name] = score

    if jm_indices:
        depth_groups = {}
        for idx in jm_indices:
            depth = depths[idx]
            if depth not in depth_groups: depth_groups[depth] = []
            depth_groups[depth].append(idx)

        for depth, indices in depth_groups.items():
            code2combo = dict()
            for i in product([0, 1], repeat=depth):
                if 1 in i:
                    code, combo = [], []
                    for j, bit in enumerate(i):
                        if bit == 1:
                            code.append(str(j))
                            combo.append(j)
                    code2combo[','.join(code)] = combo

            all_seqs = []
            mutant_codes = []
            mutant2muts = {}

            for idx in tqdm(indices, ascii=True, desc='input'):
                wt, pos_list, mt, seq_pos_list, mutation_name = parsed_mutations[idx]
                muts = [(wt[j], seq_pos_list[j], mt[j]) for j in range(depth)]
                mutant2muts[mutation_name] = muts

                base_mt_seq = mutate_seq_saprot(seq, seq_pos_list, mt)

                for code in code2combo:
                    combo = code2combo[code]
                    mask_positions = [muts[c][1] for c in combo]
                    
                    seq_wt = mask_seq_saprot(seq, mask_positions)
                    all_seqs.append(seq_wt)
                    mutant_codes.append(f'{mutation_name}|{code}|wt')
                    
                    seq_mt = mask_seq_saprot(base_mt_seq, mask_positions)
                    all_seqs.append(seq_mt)
                    mutant_codes.append(f'{mutation_name}|{code}|mt')

            code2probs_tensor = {}
            for s, mc in tqdm(list(zip(all_seqs, mutant_codes)), ascii=True, desc='jm forward'):
                tokens = tokenizer.tokenize(s)
                inputs = tokenizer(" ".join(tokens), return_tensors="pt").to(model.device)
                with torch.no_grad():
                    probs = model(**inputs).logits.softmax(dim=-1)
                code2probs_tensor[mc] = probs[0] # shape: (seq_len, vocab_size)

            all_idx = set(range(depth))
            orders = list(permutations(range(depth)))

            for idx in tqdm(indices, ascii=True, desc='output'):
                wt, pos_list, mt, seq_pos_list, mutation_name = parsed_mutations[idx]
                muts = mutant2muts[mutation_name]

                code2prob = {}
                for code in code2combo:
                    combo = code2combo[code]
                    probs_tensor_wt = code2probs_tensor[f'{mutation_name}|{code}|wt']
                    probs_tensor_mt = code2probs_tensor[f'{mutation_name}|{code}|mt']
                    for c in combo:
                        wt_aa, pos, mt_aa = muts[c]
                        ori_st = tokenizer.get_vocab()[wt_aa + foldseek_struc_vocab[0]]
                        mut_st = tokenizer.get_vocab()[mt_aa + foldseek_struc_vocab[0]]
                        code2prob[f'wt|{code}|{c}'] = probs_tensor_wt[pos + 1, ori_st: ori_st + len(foldseek_struc_vocab)].sum().item()
                        code2prob[f'mt|{code}|{c}'] = probs_tensor_mt[pos + 1, mut_st: mut_st + len(foldseek_struc_vocab)].sum().item()

                score = []
                for order in orders:
                    score_order = 0
                    for i in range(depth):
                        code = ','.join([str(idx) for idx in sorted(all_idx - set(order[:i]))])
                        c = order[i]
                        P_wt = code2prob[f'wt|{code}|{c}']
                        P_mt = code2prob[f'mt|{code}|{c}']
                        score_order += np.log(P_mt + 1e-30) - np.log(P_wt + 1e-30)
                    score.append(score_order)

                score = np.sum(score) / len(score)
                mutation_scores[mutation_name] = score
    return mutation_scores


def _score_mutations_pll_saprot(seq, parsed_mutations, model, tokenizer):
    """
    Pseudo Log-Likelihood (PLL) Scoring for SaProt.
    Score = PLL(Mutant Sequence) - PLL(Wildtype Sequence)
    PLL = sum_{i=1}^{L} log P(x_i | x_{\i})
    """
    
    def get_sequence_pll(sequence_str):
        print(sequence_str)
        print(len(sequence_str))
        tokens = tokenizer.tokenize(sequence_str)
        pll_score = 0.0
        
        for i in range(len(tokens)):
            masked_tokens = tokens[:]
            masked_tokens[i] = "#" + masked_tokens[i][-1]
            
            inputs = tokenizer(" ".join(masked_tokens), return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # logits shape: [1, seq_len+1, vocab_size]
                log_probs = torch.log_softmax(outputs.logits[0], dim=-1)

            print(log_probs.shape)
            target_idx = i + 1
            
            target_aa = tokens[i][0]
            
            
            start_st = tokenizer.get_vocab()[target_aa + foldseek_struc_vocab[0]]
            
            struct_len = len(foldseek_struc_vocab)
            slice_log_probs = log_probs[target_idx, start_st : start_st + struct_len]
            
            log_prob_aa = torch.logsumexp(slice_log_probs, dim=0).item()
            
            pll_score += log_prob_aa
            
        return pll_score

    
    # print("Calculating WT PLL...")
    wt_pll = get_sequence_pll(seq)
    
    mutation_scores = {}
    
    for wt, pos_list, mt, seq_pos_list, mutation_name in tqdm(parsed_mutations, desc='Calculating Mutant PLLs'):
        mt_seq = mutate_seq_saprot(seq, seq_pos_list, mt)
        
        mt_pll = get_sequence_pll(mt_seq)
        
        mutation_scores[mutation_name] = mt_pll - wt_pll
        
    return mutation_scores


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='SaProt scoring')
    parser.add_argument('--foldseek_bin', default="", type=str, help='Path to foldseek binary file')
    parser.add_argument('--SaProt_model_name_or_path', default="", type=str, help='Name of or path to SaProt model')
    parser.add_argument('--reference_csv', required=True, type=str, help='Path of reference CSV containing DMS_id, target_seq, pdb_file, pdb_range')
    parser.add_argument('--dms_dir', required=True, type=str, help='Path of DMS folder containing mutant CSVs')
    parser.add_argument('--structure_data_folder', required=True, type=str, help='Path of structure folder')
    parser.add_argument('--output_dir', required=True, type=str, help='Name of folder to write model scores to')
    parser.add_argument('--scoring_method', type=str, default='default', choices=['default', 'joint_marginal', 'pll', 'nopll'], help='Scoring method to use')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    model = AutoModelForMaskedLM.from_pretrained(args.SaProt_model_name_or_path, trust_remote_code=True)
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.SaProt_model_name_or_path)
    
    mapping_protein_seq_DMS = pd.read_csv(args.reference_csv)

    for _, row in mapping_protein_seq_DMS.iterrows():
        DMS_id = row["DMS_id"]
        scoring_filename = os.path.join(args.output_dir, f"{DMS_id}.csv")
        
        if os.path.exists(scoring_filename):
            print(f"Scores already computed for: {DMS_id}, skipping.")
            continue
            
        print(f"Computing scores for: {DMS_id} with method: {args.scoring_method}")
        
        target_seq = row["target_seq"].upper()
        DMS_file_name = f"{DMS_id}.csv"
        input_csv = os.path.join(args.dms_dir, DMS_file_name)
        
        if not os.path.exists(input_csv):
            print(f"DMS file not found: {input_csv}, skipping.")
            continue
            
        DMS_data = pd.read_csv(input_csv, low_memory=False)
        
        pdb_filename = row["pdb_file"]
        pdb_range = row["pdb_range"]
        
        model_scores = []
        pdb_file = os.path.join(args.structure_data_folder, pdb_filename)
        pdb_range = [int(x) for x in pdb_range.split("-")]
        target_seq_split = target_seq[pdb_range[0] - 1:pdb_range[1]]
        
        DMS_data["mutated_position"] = DMS_data['mutant'].apply(lambda x: int(x.split(':')[0][1:-1]))
        filtered_DMS_data = DMS_data[
            (DMS_data["mutated_position"] >= pdb_range[0]) & 
            (DMS_data["mutated_position"] <= pdb_range[1])
        ]
        
        if filtered_DMS_data.empty:
            continue
            
        struc_seq = get_struc_seq(args.foldseek_bin, pdb_file, ["A"], plddt_mask=True, plddt_threshold=70)["A"][1].lower()
        seq = "".join([a + b for a, b in zip(target_seq_split, struc_seq)])

        mutation_scores = score_mutations_saprot(
            seq, 
            filtered_DMS_data['mutant'].tolist(), 
            model, 
            tokenizer, 
            scoring_method=args.scoring_method
        )
        
        scores_array = np.array([mutation_scores.get(m, np.nan) for m in filtered_DMS_data['mutant']])
        model_scores.append(scores_array)
            
        if model_scores:
            model_scores = np.concatenate(model_scores)
            col = f'SaProt.{args.scoring_method}'
            DMS_data[col] = model_scores
            DMS_data[['mutant', col, 'DMS_score']].to_csv(scoring_filename, index=False)
            corr, p_value = spearmanr(DMS_data[col], DMS_data['DMS_score'])
            print(f"[{DMS_id}] Spearman Correlation: {corr:.4f} (p-value: {p_value:.2e})")
        else:
            print(f"No valid scores generated for {DMS_id}")

if __name__ == '__main__':
    main()
