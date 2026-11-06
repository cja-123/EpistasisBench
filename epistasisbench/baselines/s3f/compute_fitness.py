import os
import sys
import pprint
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdrug import core, utils, data
from torchdrug.utils import comm, pretty
from torchdrug.data import feature
from itertools import product, permutations

sys.path.append(os.path.dirname(__file__))
import util
from s3f import dataset
from s3f.model import FusionNetwork
from script.evaluate import evaluate, graph_concat, get_optimal_window

from tqdm import tqdm
import copy

METRICS = ["spearmanr", "pearsonr", "mae", "rmse"]
MODEL_WINDOW = 1022

def get_residue_prob(tasks, sequences, structure_graph, surf_graph, cfg):
    """
    Helper function to run inference on a list of proteins.
    Handles both single protein (wrapped in list) or multiple variants.
    """
    # Assuming all proteins in the list share the same structure graph reference (wild_type)
    # This is standard for mutation scoring.
    if not isinstance(sequences, list):
        protein_list = [sequences]

    _dataset = dataset.MutantDataset(sequences, structure_graph, surf_graph=surf_graph)
    
    def _func(batch_size):
        _cfg = copy.deepcopy(cfg)
        _cfg.batch_size = batch_size
        print(f'{cfg.batch_size} -> {_cfg.batch_size}', flush=True)
        return predict_with_multi_gpus(_cfg, tasks, _dataset)

    seq_probs = dynamic_batch_size(_func, cfg.batch_size)
    
    return seq_probs # logits

def mask(protein, sites, mask_id):
    masked_protein = protein.clone()
    node_index = torch.tensor(sites, dtype=torch.long)
    with masked_protein.residue():
        masked_protein.residue_type[node_index] = mask_id
        masked_protein.residue_feature[node_index] = 0
    return masked_protein

def mutate(protein, sites, muts, residue_id_map):
    mut_protein = protein.clone()
    # sites, muts : (680, 686) ('Y681N', 'K687R')
    for site, mut in zip(sites, muts):
        wt_aa, mt_aa = mut[0], mut[-1]
        mt_id = residue_id_map.get(mt_aa, -1)
        wt_id = residue_id_map.get(wt_aa, -1)
        mt_feat = feature.onehot(mt_aa, data.Protein.residue_symbol2id, allow_unknown=True)
        with mut_protein.residue():
            assert mut_protein.residue_type[site] == wt_id
            mut_protein.residue_type[site] = mt_id
            mut_protein.residue_feature[site] = torch.tensor(mt_feat)
    return mut_protein


@torch.no_grad()
def predict_with_multi_gpus(cfg, tasks, dataset):
    devices = [torch.device(i) for i in cfg.device_ids]
    output_device = devices[0]
    n_devices = len(cfg.device_ids)
    
    class ModelWrapper(nn.Module):
        def __init__(self, task):
            super().__init__()
            self.task = task
    
        def forward(self, batch):
            prob, sizes = self.task.inference(batch)
            cum_sizes = sizes.cumsum(dim=0)
            seq_prob = [prob[cum_sizes[i]-sizes[i]:cum_sizes[i]] for i in range(len(sizes))]
            return torch.stack(seq_prob)

    replicas = []
    for i in range(n_devices):
        model = ModelWrapper(task=tasks[i])
        # model = model.to(output_device)
        model = model.to(devices[i])
        model.eval()
        replicas.append(model)
    
    # replicas = nn.parallel.replicate(model, devices)

    dataloader = data.DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=0)

    seq_prob = []
    n_batch = len(dataloader)
    for i, batch in enumerate(tqdm(dataloader, ascii=True)):
        device_idx = i % n_devices
        if device_idx == 0:
            inputs = []

        device = devices[device_idx]
        batch = utils.cuda(batch, device=device)
        inputs.append(batch)
        
        if device_idx == n_devices - 1:
            outputs = nn.parallel.parallel_apply(replicas, inputs, devices=devices)
            outputs = nn.parallel.gather(outputs, output_device)
            seq_prob.append(outputs.detach().cpu())

    if inputs:
        len_inputs = len(inputs)
        replicas, devices = replicas[:len_inputs], devices[:len_inputs]
        outputs = nn.parallel.parallel_apply(replicas, inputs, devices=devices)
        outputs = nn.parallel.gather(outputs, output_device)
        seq_prob.append(outputs.detach().cpu())

    seq_prob = torch.concat(seq_prob, dim=0)
    seq_prob = list(seq_prob)
    return seq_prob

def dynamic_batch_size(func, batch_size):
    while batch_size > 0:
        try:
            return func(batch_size=batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size = batch_size // 2
                print(f'>>> OOM, now trying with batch_size = {batch_size}')
            else:
                raise # maybe other err

# ---------------------------------------------------------
# 3. Masked Marginal (a): Mask all sites simultaneously
# ---------------------------------------------------------
def score_masked_marginal_a(tasks, protein, structure_graph, surf_graph, mutations, cfg):
    residue_id_map = data.Protein.residue_symbol2id
    # mask_id = task.model.sequence_model.alphabet.get_idx("<mask>")
    mask_id = tasks[0].model.sequence_model.alphabet.get_idx("<mask>")
    
    all_masked_proteins = []
    for sites, muts, tar in mutations:
        masked_protein = mask(protein, sites, mask_id)
        all_masked_proteins.append(masked_protein)

    seq_probs = get_residue_prob(tasks, all_masked_proteins, structure_graph, surf_graph, cfg)

    scores = []
    targets = []
    for i, (sites, muts, tar) in enumerate(mutations):
        mask_logits = torch.log_softmax(seq_probs[i], dim=-1)
        item_score = 0.0
        for site, mut in zip(sites, muts):
            wt_aa, mt_aa = mut[0], mut[-1]
            wt_id = residue_id_map.get(wt_aa, -1)
            mt_id = residue_id_map.get(mt_aa, -1)
            prob_wt = mask_logits[site, wt_id]
            prob_mt = mask_logits[site, mt_id]
            item_score += (prob_mt - prob_wt).item()
        scores.append(item_score)
        targets.append(tar)
        
    return torch.tensor(scores), torch.tensor(targets)

# ---------------------------------------------------------
# 7. Joint Marginal (Epistasis for Double Mutations)
# ---------------------------------------------------------
def score_joint_marginal(tasks, protein, structure_graph, surf_graph, mutations, cfg, mean_method='Geometric', depth_threshold = 5):
    """
    Joint Marginal: Specifically for double mutations.
    Uses logic from evaluate.epi.py to generate 5 versions of sequences.
    """
    scores = []
    targets = []
    residue_id_map = data.Protein.residue_symbol2id
    # mask_id = task.model.sequence_model.alphabet.get_idx("<mask>")
    mask_id = tasks[0].model.sequence_model.alphabet.get_idx("<mask>")

    # split by depth
    mma_mutations, mma_index = [], []   # depth == 1
    jm_dict = dict()                    # 2 <= depth <= depth_threshold
    mc_dict = dict()                    # depth > depth_threshold
            
    for mutation_idx, mutation in enumerate(mutations):
        depth = len(mutation[0])
        if depth == 1:
            mma_mutations.append(mutation)
            mma_index.append(mutation_idx)
        elif depth > depth_threshold:
            if depth not in mc_dict:
                mc_dict[depth] = []
            mc_dict[depth].append((mutation_idx, mutation))
        else:
            if depth not in jm_dict:
                jm_dict[depth] = []
            jm_dict[depth].append((mutation_idx, mutation))
            
    print(f'mma: {len(mma_mutations)}, '
          f'jm: {sum(len(v) for v in jm_dict.values())}, '
          f'mc: {sum(len(v) for v in mc_dict.values())}', flush=True)
    scores_map = {}
    
    # ========== 1. mask marginal a (depth == 1) ==========
    if mma_index:
        mma_scores, mma_targets = score_masked_marginal_a(tasks, protein, structure_graph, surf_graph, mma_mutations, cfg)
        for idx, mutation_idx in enumerate(mma_index):
            scores_map[mutation_idx] = mma_scores[idx]
    
    # ========== 2. exact joint marginal (2 <= depth <= depth_threshold) ==========
    for depth in jm_dict:
        mutations_depth = jm_dict[depth]
        
        # code2combo
        # {'2': [2], '1': [1], '1,2': [1, 2], '0': [0], '0,2': [0, 2], '0,1': [0, 1], '0,1,2': [0, 1, 2]}
        code2combo = dict()
        for i in product([0, 1], repeat=depth):
            if 1 in i:
                code, combo = [], []
                for j, bit in enumerate(i):
                    if bit == 1:
                        code.append(str(j))
                        combo.append(j)
                code2combo[','.join(code)] = combo

        
        mutant_code, all_proteins, mutant2muts = [], [], dict()
        for mutation_idx, mutation in mutations_depth:
            sites, muts, tar = mutation
            mutant_str = ':'.join(muts)
            mutant2muts[mutant_str] = (sites, muts)
            
            # mut_protein
            mut_protein =  mutate(protein, sites, muts, residue_id_map)
            
            for code in code2combo:
                # combo is idx to be masked
                combo = code2combo[code]

                protein_wt = protein.clone()
                protein_mt = mut_protein.clone()
                
                mask_sites = [sites[idx] for idx in combo]

                protein_wt = mask(protein_wt, mask_sites, mask_id)
                protein_mt = mask(protein_mt, mask_sites, mask_id)

                # wt
                mutant_code.append(f'{mutant_str}|{code}|wt')
                all_proteins.append(protein_wt)
                
                # mt
                mutant_code.append(f'{mutant_str}|{code}|mt')
                all_proteins.append(protein_mt)
        
        seq_probs = get_residue_prob(tasks, all_proteins, structure_graph, surf_graph, cfg)
        seq_probs = [torch.softmax(p, dim=-1) for p in seq_probs]
        code2probs_tensor = dict(zip(mutant_code, seq_probs))
                
        all_idx = set(range(depth))
        orders = list(permutations(range(depth)))  # [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        
        joint_scores = []
        for mutation_idx, mutation in mutations_depth:
            sites, muts, tar = mutation
            mutant_str = ':'.join(muts)
            
            code2prob = dict()
            for code in code2combo:
                combo = code2combo[code]
                probs_tensor_wt = code2probs_tensor[f'{mutant_str}|{code}|wt']
                probs_tensor_mt = code2probs_tensor[f'{mutant_str}|{code}|mt']
                
                for idx in combo:
                    site = sites[idx]
                    wt_aa, mt_aa = muts[idx][0], muts[idx][-1]
                    wt_id = residue_id_map.get(wt_aa, -1)
                    mt_id = residue_id_map.get(mt_aa, -1)
                    
                    code2prob[f'wt|{code}|{idx}'] = probs_tensor_wt[site, wt_id].item()
                    code2prob[f'mt|{code}|{idx}'] = probs_tensor_mt[site, mt_id].item()
            
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
            scores_map[mutation_idx] = score

    # ========== 3. Monte Carlo estimation (depth > depth_threshold) ==========
    repeat_times = 2 ** depth_threshold  # e.g. 32 when depth_threshold=5

    for depth in mc_dict:
        mutations_depth = mc_dict[depth]

        num_mutants = len(mutations_depth)
        seeds = np.random.randint(low=0, high=1048576, size=(num_mutants, repeat_times), dtype=int)

        mutant_code, all_proteins = [], []
        mutant2muts = dict()
        mutant2code_combo = dict()

        for mutant_idx, (mutation_idx, mutation) in enumerate(mutations_depth):
            seed_list = seeds[mutant_idx]
            
            sites, muts, tar = mutation
            mutant_str = ':'.join(muts)
            mutant2muts[mutant_str] = (sites, muts)

            # make sequence_mt and sequence_wt
            mut_protein = mutate(protein, sites, muts, residue_id_map)

            mutant2code_combo[mutant_str] = []
            for repeat_idx in range(repeat_times):
                seed = seed_list[repeat_idx]
                np.random.seed(seed)
                size = np.random.randint(low=1, high=depth + 1)
                combo = sorted(np.random.choice(depth, size, replace=False).tolist())
                code = ','.join(map(str, combo))

                mutant2code_combo[mutant_str].append((code, combo))

                # mask
                mask_sites = [sites[idx] for idx in combo]
                protein_wt = mask(protein, mask_sites, mask_id)
                protein_mt = mask(mut_protein, mask_sites, mask_id)

                wt_code = f'{mutant_str}|{code}|wt'
                mt_code = f'{mutant_str}|{code}|mt'
                if wt_code not in mutant_code and mt_code not in mutant_code:
                    mutant_code.append(wt_code)
                    all_proteins.append(protein_wt)
                    mutant_code.append(mt_code)
                    all_proteins.append(protein_mt)
                else:
                    assert wt_code in mutant_code and mt_code in mutant_code

        seq_probs = get_residue_prob(tasks, all_proteins, structure_graph, surf_graph, cfg)
        seq_probs = [torch.softmax(p, dim=-1) for p in seq_probs]
        code2probs_tensor = dict(zip(mutant_code, seq_probs))

        mc_scores = []
        for mutation_idx, mutation in mutations_depth:
            sites, muts, tar = mutation
            mutant_str = ':'.join(muts)
            code_combo_list = mutant2code_combo[mutant_str]

            score = 0
            for code, combo in code_combo_list:
                size = len(combo)
                probs_tensor_wt = code2probs_tensor[f'{mutant_str}|{code}|wt']
                probs_tensor_mt = code2probs_tensor[f'{mutant_str}|{code}|mt']

                combo_score = 0
                for idx in combo:
                    site = sites[idx]
                    wt_aa, mt_aa = muts[idx][0], muts[idx][-1]
                    wt_id = residue_id_map.get(wt_aa, -1)
                    mt_id = residue_id_map.get(mt_aa, -1)

                    P_wt_idx = probs_tensor_wt[site, wt_id].item()
                    P_mt_idx = probs_tensor_mt[site, mt_id].item()
                    combo_score += np.log(P_mt_idx + 1e-30) - np.log(P_wt_idx + 1e-30)

                score += combo_score * depth / size

            score /= repeat_times
            mc_scores.append(score)

        for (mutation_idx, mutation), score in zip(mutations_depth, mc_scores):
            scores_map[mutation_idx] = score
    #--------------------------------------------------------------------------
    
    sorted_scores = [scores_map[mutation_idx] for mutation_idx in range(len(mutations))]
    sorted_targets = [mutations[mutation_idx][2] for mutation_idx in range(len(mutations))]
    
    return torch.tensor(sorted_scores), torch.tensor(sorted_targets)



# ---------------------------------------------------------
# 8. Pseudo LogLikelihood (Batch All Inputs)
# ---------------------------------------------------------
def score_pseudo_loglikelihood(tasks, protein, structure_graph, surf_graph, mutations, cfg):
    """
    Computes Pseudo-LogLikelihood scores by batching ALL inputs (WT + All Mutants) at once.
    Maximizes GPU utilization.
    """
    residue_id_map = data.Protein.residue_symbol2id
    mask_id = tasks[0].model.sequence_model.alphabet.get_idx("<mask>")
    
    seq_len = protein.num_residue
    num_mutations = len(mutations)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    all_proteins = []
    
    for i in range(seq_len):
        all_proteins.append(mask(protein, [i], mask_id))
        
    mutant_proteins = [] 
    
    for sites, muts, tar in mutations:
        mt_protein = mutate(protein, sites, muts, residue_id_map)
        mutant_proteins.append(mt_protein)
        
        for i in range(seq_len):
            all_proteins.append(mask(mt_protein, [i], mask_id))
            
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    seq_probs = get_residue_prob(tasks, all_proteins, structure_graph, surf_graph, cfg)
    
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    
    wt_ll = 0.0
    for i in range(seq_len):
        probs = torch.softmax(seq_probs[i], dim=-1)
        true_aa_id = protein.residue_type[i]
        wt_ll += torch.log(probs[i, true_aa_id] + 1e-30).item()
        
    scores = []
    targets = []
    
    for idx, (sites, muts, tar) in enumerate(mutations):
        start_idx = (idx + 1) * seq_len
        mt_prot = mutant_proteins[idx]
        
        mt_ll = 0.0
        for i in range(seq_len):
            probs = torch.softmax(seq_probs[start_idx + i], dim=-1)
            true_aa_id = mt_prot.residue_type[i]
            mt_ll += torch.log(probs[i, true_aa_id] + 1e-30).item()
            
        scores.append(mt_ll - wt_ll)
        targets.append(tar)
        
    return torch.tensor(scores), torch.tensor(targets)


# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def mutation_site(x):
    return [int(y[1:-1])-1 for y in x]


def load_dataset(csv_file):
    
    df = pd.read_csv(csv_file)
    
    mutations = []
    
    for _, row in df.iterrows():
        mut_str = row['mutant']
        score = row['DMS_score']
        
        muts = mut_str.split(":")
        sites = tuple(mutation_site(muts))
        
        if isinstance(score, str):
            tar = utils.literal_eval(score)
        else:
            tar = score
            
        mutations.append((sites, tuple(muts), tar))
    
    return mutations


def compute_fitness():
    args, vars = util.parse_args()
    # Default or ArgParse handling for scoring method
    cfg = util.load_config(args.config, context=vars)
    cfg.summary.csv_file = args.ProteinGym_reference_file
    cfg.device_ids = args.device_ids
    
    working_dir = util.create_working_directory(cfg)

    # Seed setup
    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Output dir: %s" % working_dir)
        logger.warning("Config file: %s" % args.config)
        # logger.warning("Scoring Method: %s" % args.scoring_method)
        logger.warning(pprint.pformat(cfg))

    summary = core.Configurable.load_config_dict(cfg.summary)
    id_list = cfg.get("id_list", summary.ids)

    model_name = args.model_name

    if args.DMS_index:
        DMS_id = id_list[args.DMS_index]
        print(f"Scoring {DMS_id} with model {model_name}")
        id_list = [DMS_id]

    if "exclude_id_list" in cfg:
        exclude_id_list = set(cfg.exclude_id_list)
        id_list = [id for id in id_list if id not in exclude_id_list]

    tasks = []
    for _ in range(len(cfg.device_ids)):
        task = core.Configurable.load_config_dict(cfg.task)
        task.preprocess(None, None, None)
        tasks.append(task)
        

    pd.DataFrame(columns=["DMS_id", "UniProt_ID", "seq_len", "DMS_number_single_mutants"] + METRICS).to_csv("results.csv", index=False)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))['model']
        for task in tasks:
            task.load_state_dict(model_dict)

    assay_result = {}

    # Method Dispatch Dictionary
    scoring_methods = {
        "masked_marginal_a": score_masked_marginal_a,
        "joint_marginal": score_joint_marginal,
        "pseudo_loglikelihood": score_pseudo_loglikelihood,
    }

    for scoring_method in args.scoring_method:
        if scoring_method not in scoring_methods:
            raise ValueError(f"Unknown scoring method: {scoring_method}")
        score_func = scoring_methods[scoring_method]

        for i, id in enumerate(id_list):
            DMS_filename = summary.assay_dict[id]["DMS_filename"]
            output_location = os.path.join(args.output_scores_folder, DMS_filename)
            
            csv_file = os.path.join(summary.path, DMS_filename)
            if os.path.exists(output_location):
                csv_file = output_location

                if f"{model_name}.{scoring_method}" in pd.read_csv(csv_file).columns and args.no_overwrite:
                    print(f'skiping {id} ...')
                    continue
                
            mutations = load_dataset(csv_file)
            
            if comm.get_rank() == 0:
                os.makedirs(os.path.dirname(output_location), exist_ok=True)
            
            logger.warning(pretty.separator)
            logger.warning("Start evaluation on DMS assay %s" % id)
            
            sequence = summary.assay_dict[id]["target_seq"]
            protein = data.Protein.from_sequence(sequence, atom_feature=None, bond_feature=None)
            protein.view = "residue"

            wild_types = []
            for _pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
                pdb_file = os.path.join(os.path.expanduser(cfg.structure_path), _pdb_file)
                wild_type = dataset.bio_load_pdb(pdb_file)[0]
                ca_index = wild_type.atom_name == wild_type.atom_name2id["CA"]
                wild_type = wild_type.subgraph(ca_index)
                wild_types.append(wild_type)
            wild_type = graph_concat(wild_types)
            wild_type.view = "residue"

            if cfg.get("surface_path"):
                surf_graphs = []
                res2surfs = []
                for pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
                    surf_graph_file = pdb_file.split(".")[0] + ".pkl"
                    surf_graph_file = os.path.join(os.path.expanduser(cfg.surface_path), surf_graph_file)
                    with open(surf_graph_file, "rb") as fin:
                        surf_dict = pickle.load(fin)
                        surf_graph = dataset.load_surface(surf_dict)
                        surf_graphs.append(surf_graph)
                        res2surfs.append(torch.as_tensor(surf_dict["res2surf"]))
                surf_graph = graph_concat(surf_graphs)
                res2surf = torch.cat(res2surfs, dim=0)
                with wild_type.residue():
                    wild_type.res2surf = res2surf
            else:
                surf_graph = None
    
            pdb_range = summary.assay_dict[id]["pdb_range"].split("-")
            start, end = int(pdb_range[0])-1, int(pdb_range[-1])
            with wild_type.graph():
                wild_type.start = torch.as_tensor(start)
                wild_type.end = torch.as_tensor(end)
                
            # if id=="POLG_HCVJF_Qi_2014": start, end = 1981, 2225
            if protein.num_residue > MODEL_WINDOW and DMS_filename not in ["POLG_HCVJF_Qi_2014.csv", "A0A140D2T1_ZIKV_Sourisseau_2019.csv", "B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv"]:
                logger.warning(f"Sequence length {protein.num_residue} > {MODEL_WINDOW}. Switching to per-mutation sliding window mode.")
                
                all_preds = []
                all_targets = []
                
                for i, mutation in enumerate(mutations):
                    sites, muts, tar = mutation
                    
                    start, end = get_optimal_window(
                        mutation_position_relative=sites[0],
                        seq_len_wo_special=protein.num_residue,
                        model_window=MODEL_WINDOW
                    )
                    if DMS_filename == "BRCA2_HUMAN_Erwood_2022_HEK293T.csv": #Truncate sequence window to align with truncated structure
                        if end > 2832:
                            start, end = 1820, 2832
                            
                    residue_mask = torch.zeros((protein.num_residue, ), dtype=torch.bool)
                    residue_mask[start:end] = 1
                    sliced_protein = protein.subresidue(residue_mask)
                    with sliced_protein.graph():
                        sliced_protein.start = torch.as_tensor(start)
                        sliced_protein.end = torch.as_tensor(end)
                        
                    adjusted_sites = tuple(s - start for s in sites)
                    adjusted_mutation = (adjusted_sites, muts, tar)
                    
                    pred, target = score_func(tasks, sliced_protein, wild_type, surf_graph, [adjusted_mutation], cfg)
                    
                    all_preds.append(pred)
                    all_targets.append(target)
                    
                if all_preds:
                    pred = torch.cat(all_preds)
                    target = torch.cat(all_targets)
                else:
                    pred, target = torch.tensor([]), torch.tensor([])
            else:
                # -------------------------------------
                if DMS_filename == "POLG_HCVJF_Qi_2014.csv":
                    start, end = 1981, 2225
                elif DMS_filename == "A0A140D2T1_ZIKV_Sourisseau_2019.csv":
                    start, end = 290, 794
                elif DMS_filename == "B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv":
                    start, end = 119, 197  # keep high plddt part
                else:
                    start, end = 0, protein.num_residue

                # -------------------------------------
    

    
                residue_mask = torch.zeros((protein.num_residue, ), dtype=torch.bool)
                residue_mask[start:end] = 1
                protein = protein.subresidue(residue_mask)
                
                adjusted_mutations = []
                for sites, muts, tar in mutations:
                    if all(start <= s < end for s in sites):
                        adjusted_sites = tuple(s - start for s in sites)
                        adjusted_mutations.append((adjusted_sites, muts, tar))
                    else:
                        if comm.get_rank() == 0:
                            logger.warning(f"Mutation {muts} is outside the structure range [{start}, {end})")
                
                mutations = adjusted_mutations
                
                with protein.graph():
                    protein.start = torch.as_tensor(start)
                    protein.end = torch.as_tensor(end)
                
                if comm.get_rank() == 0:
                    logger.warning("Number of mutations: %d" % len(mutations))

                pred, target = [], []
                chunk_size = args.mutant_chunk_size
                for i in range(0, len(mutations), chunk_size):
                    _mutations = mutations[i: i + chunk_size]
                    _pred, _target = score_func(tasks, protein, wild_type, surf_graph, _mutations, cfg)
                    pred.append(_pred), target.append(_target)
                pred, target = torch.concat(pred, dim=0), torch.concat(target, dim=0)
                
            result = evaluate(pred, target)

            if comm.get_rank() == 0:
                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()
                
                mut_str_list = [":".join(m[1]) for m in mutations]
                

                mutant2score = dict(zip(mut_str_list, pred_np))
                output_df = pd.read_csv(csv_file)
                output_df[f"{model_name}.{scoring_method}"] = output_df.mutant.map(mutant2score)
                assert not output_df.isna().any(axis=None).item()
                
                output_df.to_csv(output_location, index=False)
                
                logger.warning(pretty.separator)
                logger.warning("Test results")
                logger.warning(pretty.line)
                logger.warning(pprint.pformat(result))

                util.write_aggregate_performance(args.output_scores_folder, id, summary, result, METRICS, result_filename=f"results_{model_name}_{scoring_method}.csv")
                assay_result[id] = result

    if comm.get_rank() == 0:
        logger.warning(pretty.separator)
        logger.warning("Average results on all assays")
        logger.warning(pretty.line)
        logger.warning(pprint.pformat(utils.mean(utils.stack(list(assay_result.values())))))

if __name__ == "__main__":
    compute_fitness()
