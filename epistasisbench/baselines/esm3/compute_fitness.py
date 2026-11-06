import torch
from tqdm import tqdm
import numpy as np
import re
import os
import pandas as pd
from scipy.stats import spearmanr
import attr
import warnings
from Bio.PDB import PDBParser, Selection
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Import both model types
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.sdk.api import ESMProteinTensor
from itertools import product, permutations

from joblib import Parallel, delayed
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mask_sequence(protein: ESMProtein, positions: list[int]) -> ESMProtein:
    if protein.sequence is None:
        raise ValueError("Protein must have a sequence to mask")
    
    masked_sequence = list(protein.sequence)
    for pos in positions:
        masked_sequence[pos] = '_'
    
    return ESMProtein(
        sequence=''.join(masked_sequence),
        coordinates=protein.coordinates,
        secondary_structure=protein.secondary_structure,
        sasa=protein.sasa,
        function_annotations=protein.function_annotations
    )

def mutate_sequence(protein: ESMProtein, positions: list[int], mutations: list[str], wildtypes) -> ESMProtein:
    if protein.sequence is None:
        raise ValueError("Protein must have a sequence to mutate")
    
    mutated_sequence = list(protein.sequence)
    
    # for pos, mut in zip(positions, mutations):
    #     wt_aa, mt_aa = mut[0], mut[-1]
    
    for pos, wt_aa, mt_aa in zip(positions, wildtypes, mutations):
        if mutated_sequence[pos] != wt_aa:
            print(f"Warning: Wildtype mismatch at position {pos}: expected {wt_aa}, got {mutated_sequence[pos]}")
        mutated_sequence[pos] = mt_aa
    
    return ESMProtein(
        sequence=''.join(mutated_sequence),
        coordinates=protein.coordinates,
        secondary_structure=protein.secondary_structure,
        sasa=protein.sasa,
        function_annotations=protein.function_annotations
    )
    
def model_to_multi_devices(model, devices):
    device = torch.device(devices[0])
    device_ids = devices
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device)
    return model
        
# ----------------------------------------------------------------------------------------------


def score_mutations(sequence, mutations, model=None, model_type="esmc_300M", model_path=None, window_size=1024, devices=[0]):
    """
    Score mutations using the masked-marginals approach with sequence-only ESM models.

    Args:
        sequence (str): Protein sequence
        mutations (list): List of mutations in format "A25G" (wt, position, mutant)
        model: ESM model instance (optional, will load if not provided)
        model_type (str): Type of model to use ('esmc_300M', 'esmc_600M', or 'esm3_open')
        model_path (str, optional): Path to local model checkpoint (if None, downloads from HuggingFace)
        window_size (int): Size of window for long sequences

    Returns:
        dict: Dictionary of mutation scores
    """
    # Load model if not provided
    if model is None:
        if model_type == "esmc_300M":
            if model_path:
                print(f"Loading ESMC_300M from local path: {model_path}")
                model = ESMC.from_pretrained(model_path).to("cuda")
            else:
                print("Downloading ESMC_300M from HuggingFace")
                model = ESMC.from_pretrained("esmc_300m").to("cuda")
        elif model_type == "esmc_600M":
            if model_path:
                print(f"Loading ESMC_600M from local path: {model_path}")
                model = ESMC.from_pretrained(model_path).to("cuda")
            else:
                print("Downloading ESMC_600M from HuggingFace")
                model = ESMC.from_pretrained("esmc_600m").to("cuda")
        elif model_type == "esm3_open":
            if model_path:
                print(f"Loading ESM3_open from local path: {model_path}")
                model = ESM3.from_pretrained(model_path).to("cuda")
            else:
                print("Downloading ESM3_open from HuggingFace")
                model = ESM3.from_pretrained("esm3-open").to("cuda")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Make sure we're in eval mode
    raise
    # model = model_to_multi_devices(model, devices)
    model.eval()

    if len(sequence) == 0:
        raise ValueError("Empty sequence provided")

    print(f"Working with sequence of length {len(sequence)} using {model_type} model")

    # Parse mutations and validate them against the sequence
    parsed_mutations = []
    for mutation in mutations:
        # Handle multiple mutations separated by colon
        if ":" in mutation:
            sub_mutations = mutation.split(":")
            multi_wt, multi_mt = "", ""
            multi_pos = []
            multi_seq_pos = []
            valid_multi = True

            for sub_mutation in sub_mutations:
                # Parse single mutation format (e.g., "A25G")
                match = re.match(r"([A-Z])(\d+)([A-Z])", sub_mutation)
                if not match:
                    print(f"Warning: Could not parse mutation {sub_mutation}, skipping")
                    valid_multi = False
                    break

                wt, pos_str, mt = match.groups()
                pos = int(pos_str)
                # For sequence-only, positions are typically 1-indexed in mutation notation
                # but we need 0-indexed for array access
                seq_pos = pos - 1

                # Check if position is valid
                if seq_pos < 0 or seq_pos >= len(sequence):
                    print(f"Warning: Position {pos} out of range (sequence length: {len(sequence)}), skipping")
                    valid_multi = False
                    break

                # Check if wildtype matches
                if sequence[seq_pos] != wt:
                    print(f"Warning: Wild-type {wt} at position {pos} doesn't match sequence {sequence[seq_pos]} at index {seq_pos}, skipping")
                    valid_multi = False
                    break

                multi_wt += wt
                multi_mt += mt
                multi_pos.append(pos)
                multi_seq_pos.append(seq_pos)

            if valid_multi:
                # Add combined mutation
                parsed_mutations.append((multi_wt, multi_pos, multi_mt, multi_seq_pos, mutation))
        else:
            # Parse single mutation format (e.g., "A25G")
            match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
            if not match:
                print(f"Warning: Could not parse mutation {mutation}, skipping")
                continue

            wt, pos_str, mt = match.groups()
            pos = int(pos_str)
            # Convert to 0-indexed
            seq_pos = pos - 1

            # Check if position is valid
            if seq_pos < 0 or seq_pos >= len(sequence):
                print(f"Warning: Position {pos} out of range (sequence length: {len(sequence)}), skipping")
                continue

            # Check if wildtype matches
            if sequence[seq_pos] != wt:
                print(f"Warning: Wild-type {wt} at position {pos} doesn't match sequence {sequence[seq_pos]} at index {seq_pos}, skipping")
                continue

            parsed_mutations.append((wt, [pos], mt, [seq_pos], mutation))

    # Create protein object
    protein = ESMProtein(sequence=sequence)
    
    # Use common scoring function
    return _score_mutations_common(sequence, parsed_mutations, protein, model, window_size)


def score_mutations_with_pdb(pdb_path, mutations, chain_id=None, model=None, model_type="esm3_open", model_path=None, use_structure=True, window_size=1024, pdb_range=None, scoring_method='default', devices=[0], batch_size=1):
    """
    Score mutations using the masked-marginals approach with ESM3 using structure from PDB.
    Only used with ESM3 model.

    Args:
        pdb_path (str): Path to the PDB file
        mutations (list): List of mutations in format "A25G" (wt, position, mutant)
        chain_id (str, optional): Chain ID to use from PDB. If None, uses first chain.
        model: ESM3 model (optional, will load if not provided)
        model_type (str): Type of model to use (should be "esm3_open" for this function)
        model_path (str, optional): Path to local model checkpoint (if None, downloads from HuggingFace)
        use_structure (bool): Whether to use structure information or sequence-only
        window_size (int): Size of window for long sequences
        pdb_range (str, optional): Range of residues in the sequence covered by the PDB (format: "start-end", 1-indexed)

    Returns:
        dict: Dictionary of mutation scores
    """
    if model_type != "esm3_open":
        raise ValueError("score_mutations_with_pdb only supports ESM3_open model")
        
    # Load model if not provided
    if model is None:
        if model_path:
            print(f"Loading ESM3_open from local path: {model_path}")
            model = ESM3.from_pretrained(model_path).to("cuda")
        else:
            print("Downloading ESM3_open from HuggingFace")
            model = ESM3.from_pretrained("esm3-open").to("cuda")

    # modified
    model = model_to_multi_devices(model, devices)
    
    # Make sure we're in eval mode
    model.eval()

    # Load PDB file using ProteinChain
    protein_chain = ProteinChain.from_pdb(pdb_path, chain_id=chain_id)
    
    # Create ESMProtein from ProteinChain with explicit coordinate setting
    if use_structure:
        protein = ESMProtein(
            sequence=protein_chain.sequence,
            coordinates=torch.tensor(protein_chain.atom37_positions)
        )
        # Verify coordinates exist and print shape
        print(f"Coordinates tensor shape: {protein.coordinates.shape}")
    else:
        # If not using structure, just use the sequence
        protein = ESMProtein(sequence=protein_chain.sequence)

    sequence = protein_chain.sequence

    # Handle pdb_range if provided
    pdb_offset = 0
    if pdb_range:
        try:
            start, end = map(int, pdb_range.split('-'))
            # Convert from 1-indexed to 0-indexed
            pdb_offset = start - 1
            
            # Print info about the PDB range
            print(f"PDB structure covers residues {start}-{end} (1-indexed)")
            if pdb_offset > 0:
                print(f"Using offset of {pdb_offset} to align PDB residues with sequence")
        except:
            print(f"Warning: Could not parse pdb_range '{pdb_range}', using no offset")

    if len(sequence) == 0:
        raise ValueError("Could not extract valid sequence from PDB file")

    print(f"Extracted sequence of length {len(sequence)} from PDB")

    # Create a mapping from PDB residue numbers to sequence indices
    # Using residue_index from ProteinChain
    residue_id_to_idx = {}
    for i, residue_id in enumerate(protein_chain.residue_index):
        # Handle insertion codes if present
        if hasattr(protein_chain, 'insertion_code') and protein_chain.insertion_code is not None:
            ins_code = protein_chain.insertion_code[i]
            if ins_code and ins_code.strip():  # If there's a non-empty insertion code
                residue_id = f"{residue_id}{ins_code}"

        # Apply the offset from pdb_range to map to the correct position in target sequence
        residue_id_to_idx[int(residue_id) + pdb_offset] = i

    # Parse mutations and validate them against the sequence
    parsed_mutations = []
    for mutation in mutations:
        # Handle multiple mutations separated by colon
        if ":" in mutation:
            sub_mutations = mutation.split(":")
            multi_wt, multi_mt = "", ""
            multi_pos = []
            multi_seq_pos = []
            valid_multi = True

            for sub_mutation in sub_mutations:
                # Parse single mutation format (e.g., "A25G")
                match = re.match(r"([A-Z])(\d+)([A-Z])", sub_mutation)
                if not match:
                    print(f"Warning: Could not parse mutation {sub_mutation}, skipping")
                    valid_multi = False
                    break

                wt, pos_str, mt = match.groups()
                pos = int(pos_str)

                # Check if position exists in PDB
                if pos not in residue_id_to_idx:
                    print(f"Warning: Position {pos} not found in PDB, skipping")
                    valid_multi = False
                    break

                # Map PDB position to sequence index
                seq_pos = residue_id_to_idx[pos]

                multi_wt += wt
                multi_mt += mt
                multi_pos.append(pos)
                multi_seq_pos.append(seq_pos)

            if valid_multi:
                # Add combined mutation
                parsed_mutations.append((multi_wt, multi_pos, multi_mt, multi_seq_pos, mutation))
        else:
            # Parse single mutation format (e.g., "A25G")
            match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
            if not match:
                print(f"Warning: Could not parse mutation {mutation}, skipping")
                continue

            wt, pos_str, mt = match.groups()
            pos = int(pos_str)

            # Check if position exists in PDB
            if pos not in residue_id_to_idx:
                print(f"Warning: Position {pos} not found in PDB, skipping")
                continue

            # Map PDB position to sequence index
            seq_pos = residue_id_to_idx[pos]

            parsed_mutations.append((wt, [pos], mt, [seq_pos], mutation))

    # Get the rest of the scoring from the common function
    if scoring_method == 'joint_marginal':
        return _score_mutations_joint(sequence, parsed_mutations, protein, model, window_size, batch_size=batch_size)
    elif scoring_method == 'mask_marginal':
        return _score_mutations_mask_marginal(sequence, parsed_mutations, protein, model, window_size, batch_size=batch_size)
    elif scoring_method == 'pseudo_loglikelihood':
        return _score_mutations_pll(sequence, parsed_mutations, protein, model, window_size, batch_size=batch_size)
    elif scoring_method == 'default':
        return _score_mutations_common(sequence, parsed_mutations, protein, model, window_size)
    else:
        print(scoring_method)
        raise


def _score_mutations_common(sequence, parsed_mutations, protein, model, window_size=1024):
    """
    Common code for scoring mutations with both ESM-C and ESM3 models.
    
    Args:
        sequence (str): Protein sequence
        parsed_mutations (list): List of parsed mutations
        protein (ESMProtein): Encoded protein
        model: ESM model (ESM3 or ESM-C)
        window_size (int): Size of window for long sequences
        
    Returns:
        dict: Dictionary of mutation scores
    """
    if not parsed_mutations:
        print("No valid mutations to score")
        return {}
        
    # Map amino acids to tokens
    aa_to_token = {}
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    # Build a mapping of amino acids to token IDs
    for aa in amino_acids:
        encoded_protein = ESMProtein(sequence=aa)
        tensor = model.encode(encoded_protein)
        # Get the token ID (skipping special tokens)
        aa_to_token[aa] = tensor.sequence[1].item()

    # Encode the protein to get tensor representation
    protein_tensor = model.encode(protein)

    # Get sequence tokens
    sequence_tokens = protein_tensor.sequence

    # Store mutation scores
    mutation_scores = {}

    # For each position, mask it and get the logits
    # Map from sequence position to position in all_token_probs
    position_map = {}
    all_token_probs = []

    # Get all unique positions that need to be scored
    all_positions = []
    for _, _, _, seq_positions, _ in parsed_mutations:
        if isinstance(seq_positions, list):
            all_positions.extend(seq_positions)
        else:
            all_positions.append(seq_positions)
    positions_to_score = sorted(set(all_positions))

    # Check if sequence is too long for model context
    seq_len = len(sequence)
    needs_windowing = seq_len > window_size - 2  # Account for special tokens

    if needs_windowing:
        print(f"Sequence length {seq_len} exceeds model window size {window_size-2}, using windowing approach")

    # Get mask token ID based on model type
    # For ESM-C, mask token is typically 32
    mask_token_id = getattr(model, "mask_token_id", 32)
    
    # If using ESM3, get mask token ID from tokenizers
    if isinstance(model, ESM3) and hasattr(model, "tokenizers"):
        mask_token_id = model.tokenizers.sequence.mask_token_id

    for i, seq_pos in enumerate(tqdm(positions_to_score, desc="Scoring mutations")):
        # Map the sequence position to the index in all_token_probs
        position_map[seq_pos] = i

        # Calculate token position (add 1 for BOS token)
        token_pos = seq_pos + 1

        # Handle long sequences with windowing
        if needs_windowing:
            # Determine optimal window around mutation position
            window_half = (window_size - 2) // 2  # Excluding special tokens
            start_pos = max(0, seq_pos - window_half)
            end_pos = min(seq_len, start_pos + window_size - 2)

            # Adjust start if end is at sequence boundary
            if end_pos == seq_len:
                start_pos = max(0, seq_len - (window_size - 2))

            # Get the window
            window_seq = sequence[start_pos:end_pos]

            # Create new protein for the window
            if hasattr(protein, 'coordinates') and protein.coordinates is not None:
                # Extract the relevant part of coordinates if using structure
                window_protein = ESMProtein(
                    sequence=window_seq,
                    coordinates=protein.coordinates[start_pos:end_pos]
                )
            else:
                window_protein = ESMProtein(sequence=window_seq)

            # Encode window
            window_tensor = model.encode(window_protein)

            # Calculate new position in window
            window_token_pos = seq_pos - start_pos + 1  # +1 for BOS

            # Clone tokens and apply mask
            masked_tokens = window_tensor.sequence.clone()
            masked_tokens[window_token_pos] = mask_token_id

            # Create masked protein tensor
            masked_protein_tensor = attr.evolve(window_tensor, sequence=masked_tokens)
        else:
            # Clone tokens and apply mask for full sequence
            masked_tokens = sequence_tokens.clone()
            masked_tokens[token_pos] = mask_token_id

            # Create a new protein tensor with masked tokens
            masked_protein_tensor = attr.evolve(protein_tensor, sequence=masked_tokens)
            window_token_pos = token_pos  # Use original position

        # Get logits at the masked position
        with torch.no_grad():
            logits_output = model.logits(
                masked_protein_tensor,
                LogitsConfig(sequence=True)
            )

        # Calculate log probabilities
        if needs_windowing:
            token_logits = logits_output.logits.sequence[0, window_token_pos]
        else:
            token_logits = logits_output.logits.sequence[0, token_pos]

        token_probs = torch.log_softmax(token_logits, dim=-1)
        all_token_probs.append(token_probs)

    # Score mutations
    for wt, pos_list, mt, seq_pos_list, mutation_name in parsed_mutations:
        # Calculate score based on substitution approach
        score = 0.0

        # Handle multiple mutations
        if len(seq_pos_list) > 1:
            for i, seq_pos in enumerate(seq_pos_list):
                # Get single wt and mt for this position
                single_wt = wt[i]
                single_mt = mt[i]

                # Get token indices for amino acids
                wt_idx = aa_to_token[single_wt]
                mt_idx = aa_to_token[single_mt]

                # Get the index in all_token_probs
                probs_idx = position_map[seq_pos]

                # Extract probabilities
                token_probs = all_token_probs[probs_idx]

                # Add to cumulative score (mt_prob - wt_prob)
                score += (token_probs[mt_idx] - token_probs[wt_idx]).item()
        else:
            # Handle single mutation
            if isinstance(seq_pos_list, list):
                seq_pos = seq_pos_list[0]
            else:
                seq_pos = seq_pos_list

            # Get token indices for amino acids
            wt_idx = aa_to_token[wt]
            mt_idx = aa_to_token[mt]

            # Get the index in all_token_probs
            probs_idx = position_map[seq_pos]

            # Extract probabilities
            token_probs = all_token_probs[probs_idx]

            # Compute score (mt_prob - wt_prob)
            score = (token_probs[mt_idx] - token_probs[wt_idx]).item()

        # Store score
        mutation_scores[mutation_name] = score

    return mutation_scores


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
def batch_generator(ESMProtein_list, batch_size):
    def stack_track(ESMProtein_list, track_name):
        tensors = [getattr(p, track_name) for p in ESMProtein_list]
        if all(t is None for t in tensors):
            return None
        return torch.stack(tensors, dim=0)

    # print(batch_size, len(ESMProtein_list))
    for left in tqdm(range(0, len(ESMProtein_list), batch_size), ascii=True, desc='batch forward'):
        right = left + batch_size
        batch =  ESMProtein_list[left:right]
        batch = _BatchedESMProteinTensor(
            sequence=stack_track(batch, "sequence"),
            structure=stack_track(batch, "structure"),
            secondary_structure=stack_track(batch, "secondary_structure"),
            sasa=stack_track(batch, "sasa"),
            function=stack_track(batch, "function"),
            residue_annotations=stack_track(batch, "residue_annotations"),
            coordinates=stack_track(batch, "coordinates"),
        )
        yield batch


def _score_mutations_joint(
    sequence, 
    parsed_mutations, 
    protein, 
    model, 
    window_size=1024,
    depth_threshold=5,
    batch_size=1
):
    if not parsed_mutations:
        print("No valid mutations to score")
        return {}
    
    depths = [len(seq_pos_list) if isinstance(seq_pos_list, list) else 1 
              for _, _, _, seq_pos_list, _ in parsed_mutations]
    
    mma_indices = [i for i, depth in enumerate(depths) if depth == 1 or depth > depth_threshold]
    jm_indices = [i for i, depth in enumerate(depths) if 2 <= depth <= depth_threshold]
    
    mutation_scores = {}
    
    aa_to_token = {}
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    for aa in amino_acids:
        encoded_protein = ESMProtein(sequence=aa)
        tensor = model.encode(encoded_protein)
        aa_to_token[aa] = tensor.sequence[1].item()
    
    # mask_token_id = model.tokenizers.sequence.mask_token_id
    
    if mma_indices:
        print(f"Processing {len(mma_indices)} mutations with masked marginal a")
        all_mask_positions = set()
        for idx in mma_indices:
            _, _, _, seq_pos_list, _ = parsed_mutations[idx]
            all_mask_positions.update(seq_pos_list)
        
        masked_proteins = []
        for idx in mma_indices:
            wt, pos_list, mt, seq_pos_list, mutation_name = parsed_mutations[idx]
            # masked_protein = mask_sequence(protein, seq_pos_list, mask_token_id)
            masked_protein = mask_sequence(protein, seq_pos_list)
            masked_proteins.append(masked_protein)

        masked_proteins = [model.encode(i) for i in masked_proteins]
        logits_output = []
        for batch in batch_generator(masked_proteins, batch_size):
            logits = model.logits(batch, LogitsConfig(sequence=True))
            logits_output.append(logits.logits.sequence)
            
        logits_output = torch.concat(logits_output, dim=0)
        
        for i, idx in enumerate(mma_indices):
            wt, pos_list, mt, seq_pos_list, mutation_name = parsed_mutations[idx]
            score = 0.0
            
            for j, seq_pos in enumerate(seq_pos_list):
                single_wt = wt[j]
                single_mt = mt[j]
                
                token_probs = torch.log_softmax(
                    logits_output[i, seq_pos + 1], dim=-1
                )
                wt_idx = aa_to_token[single_wt]
                mt_idx = aa_to_token[single_mt]
                
                score += (token_probs[mt_idx] - token_probs[wt_idx]).item()
            
            mutation_scores[mutation_name] = score
    
    if jm_indices:
        print(f"Processing {len(jm_indices)} mutations with joint marginal")
        depth_groups = {}
        for idx in jm_indices:
            depth = depths[idx]
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(idx)
        
        for depth, indices in depth_groups.items():
            print(f"Processing depth {depth} with {len(indices)} mutations")
            code2combo = dict()
            for i in product([0, 1], repeat=depth):
                if 1 in i:
                    code, combo = [], []
                    for j, bit in enumerate(i):
                        if bit == 1:
                            code.append(str(j))
                            combo.append(j)
                    code2combo[','.join(code)] = combo
            
            all_proteins = []
            mutant_codes = []
            mutant2muts = {}
            
            for idx in tqdm(indices, ascii=True, desc='input'):
                wt, pos_list, mt, seq_pos_list, mutation_name = parsed_mutations[idx]
                muts = [(wt[j], seq_pos_list[j], mt[j]) for j in range(depth)]
                mutant2muts[mutation_name] = muts
                
                # print(parsed_mutations[idx])
                # print(seq_pos_list)
                # print(mt)
                # print(protein)
                # raise
                base_mt_protein = mutate_sequence(protein, seq_pos_list, mt, wt)
                
                for code in code2combo:
                    combo = code2combo[code]
                    
                    mask_positions = [muts[c][1] for c in combo]
                    
                    # protein_wt = mask_sequence(protein, mask_positions, mask_token_id)
                    protein_wt = mask_sequence(protein, mask_positions)
                    all_proteins.append(protein_wt)
                    mutant_codes.append(f'{mutation_name}|{code}|wt')
                    
                    # protein_mt = mask_sequence(base_mt_protein, mask_positions, mask_token_id)
                    protein_mt = mask_sequence(base_mt_protein, mask_positions)
                    all_proteins.append(protein_mt)
                    mutant_codes.append(f'{mutation_name}|{code}|mt')

            # forward
            all_proteins = [model.encode(i) for i in tqdm(all_proteins, ascii=True, desc='encode')]
            # all_proteins = Parallel(n_jobs=40, verbose=10)(
            #     delayed(model.encode)(i) for i in tqdm(all_proteins, ascii=True, desc='encode')
            # )
            logits_output = []
            # for batch in tqdm(batch_generator(all_proteins, batch_size), ascii=True, desc='forward'):
            print(56565656, batch_size)
            for batch in batch_generator(all_proteins, batch_size):
                logits = model.logits(batch, LogitsConfig(sequence=True))
                logits_output.append(logits.logits.sequence)
                
            logits_output = torch.concat(logits_output, dim=0)
            
            probs = torch.softmax(logits_output, dim=-1)
            code2probs_tensor = dict(zip(mutant_codes, list(probs)))
            
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
                        wt_idx = aa_to_token[wt_aa]
                        mt_idx = aa_to_token[mt_aa]
                        code2prob[f'wt|{code}|{c}'] = probs_tensor_wt[pos + 1, wt_idx].item()
                        code2prob[f'mt|{code}|{c}'] = probs_tensor_mt[pos + 1, mt_idx].item()
                
                # P_wt, P_mt = [], []
                # for order in orders:
                #     P_wt_order, P_mt_order = 1, 1
                #     for i in range(depth):
                #         code = ','.join([str(idx) for idx in sorted(all_idx - set(order[:i]))])
                #         c = order[i]
                #         P_wt_order *= code2prob[f'wt|{code}|{c}']
                #         P_mt_order *= code2prob[f'mt|{code}|{c}']
                #     P_wt.append(P_wt_order)
                #     P_mt.append(P_mt_order)
                
                # score = np.log(np.sum(P_mt) + 1e-30) - np.log(np.sum(P_wt) + 1e-30)
                # mutation_scores[mutation_name] = score
                
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



def _calculate_sequence_pll(protein: ESMProtein, model, aa_to_token: dict, batch_size: int = 1):
    sequence = protein.sequence
    seq_len = len(sequence)
    if seq_len == 0:
        return 0.0

    masked_proteins = []
    for i in range(seq_len):
        masked_proteins.append(mask_sequence(protein, [i]))

    encoded_proteins = [model.encode(p) for p in masked_proteins]

    all_logits = []
    for batch in batch_generator(encoded_proteins, batch_size):
        output = model.logits(batch, LogitsConfig(sequence=True))
        all_logits.append(output.logits.sequence)
    
    logits_tensor = torch.concat(all_logits, dim=0)
    
    total_pll = 0.0
    
    log_probs = torch.log_softmax(logits_tensor, dim=-1)
    
    for i in range(seq_len):
        
        token_idx = i + 1
        
        if token_idx >= logits_tensor.shape[1]:
            continue
            
        pos_log_probs = log_probs[i, token_idx]
        
        original_aa = sequence[i]
        if original_aa not in aa_to_token:
            continue
            
        aa_id = aa_to_token[original_aa]
        total_pll += pos_log_probs[aa_id].item()
        
    return total_pll

def _score_mutations_pll(sequence, parsed_mutations, protein, model, window_size=1024, batch_size=1):
    if not parsed_mutations:
        print("No valid mutations to score")
        return {}

    aa_to_token = {}
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    for aa in amino_acids:
        encoded_protein = ESMProtein(sequence=aa)
        tensor = model.encode(encoded_protein)
        aa_to_token[aa] = tensor.sequence[1].item()

    print("Calculating Wild-Type PLL...")
    wt_pll = _calculate_sequence_pll(protein, model, aa_to_token, batch_size)
    print(f"Wild-Type PLL: {wt_pll:.4f}")

    mutation_scores = {}
    
    for wt, pos_list, mt, seq_pos_list, mutation_name in tqdm(parsed_mutations, desc="Scoring mutations (PLL)"):
        mutant_protein = mutate_sequence(protein, seq_pos_list, list(mt), list(wt))
        
        mt_pll = _calculate_sequence_pll(mutant_protein, model, aa_to_token, batch_size)
        
        score = mt_pll - wt_pll
        mutation_scores[mutation_name] = score

    return mutation_scores




def _score_mutations_mask_marginal(
    sequence, 
    parsed_mutations, 
    protein, 
    model, 
    window_size=1024, 
    batch_size=1
):
    """
    Score mutations using the mask_marginal approach with structure context.
    For each mutation (including multi-point mutations), mask all mutated positions 
    simultaneously, predict probabilities, and sum the log-odds scores.
    
    Args:
        sequence (str): Protein sequence
        parsed_mutations (list): List of parsed mutations
        protein (ESMProtein): Encoded protein with coordinates
        model: ESM3 model
        window_size (int): Window size (passed for consistency, windowing logic follows model limits)
        batch_size (int): Batch size for inference
    Returns:
        dict: Dictionary of mutation scores
    """
    if not parsed_mutations:
        print("No valid mutations to score")
        return {}

    aa_to_token = {}
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    for aa in amino_acids:
        encoded_protein = ESMProtein(sequence=aa)
        tensor = model.encode(encoded_protein)
        aa_to_token[aa] = tensor.sequence[1].item()  # 0 is BOS, 1 is the AA token

    all_proteins = []
    mutation_metadata = [] 
    
    print(f"Preparing {len(parsed_mutations)} masked proteins for mask_marginal scoring...")
    
    for wt, pos_list, mt, seq_pos_list, mutation_name in parsed_mutations:
        masked_protein = mask_sequence(protein, seq_pos_list)
        all_proteins.append(masked_protein)
        
        mutation_metadata.append({
            'wt': wt, 
            'mt': mt, 
            'seq_pos_list': seq_pos_list, 
            'mutation_name': mutation_name
        })

    encoded_proteins = [model.encode(p) for p in tqdm(all_proteins, desc="Encoding proteins")]
    
    logits_outputs = []
    for batch in batch_generator(encoded_proteins, batch_size):
        logits = model.logits(batch, LogitsConfig(sequence=True))
        logits_outputs.append(logits.logits.sequence)
    
    logits_tensor = torch.concat(logits_outputs, dim=0)

    mutation_scores = {}
    
    for i, meta in enumerate(mutation_metadata):
        wt_str = meta['wt']
        mt_str = meta['mt']
        seq_positions = meta['seq_pos_list']
        name = meta['mutation_name']
        
        # logits_tensor[i] shape: [SeqLen, Vocab]
        current_logits = logits_tensor[i]
        
        log_probs = torch.log_softmax(current_logits, dim=-1)
        
        score = 0.0
        
        for j, seq_pos in enumerate(seq_positions):
            single_wt = wt_str[j]
            single_mt = mt_str[j]
            
            token_pos = seq_pos + 1
            
            wt_idx = aa_to_token[single_wt]
            mt_idx = aa_to_token[single_mt]
            
            # log_probs shape: [SeqLen, Vocab]
            log_p_wt = log_probs[token_pos, wt_idx].item()
            log_p_mt = log_probs[token_pos, mt_idx].item()
            
            score += (log_p_mt - log_p_wt)
        
        mutation_scores[name] = score
        
    return mutation_scores



# ----------------------------------------------------------------------------------------------


def process_csv_and_score_mutations(csv_path, model_type="esmc_300M", model_path=None, pdb_path=None, sequence_file=None, sequence=None, chain_id=None, use_structure=True, pdb_range=None, output_path=None, scoring_method='default', devices=[0], batch_size=1):
    """
    Process a CSV file with mutations and calculate Spearman correlation with DMS scores.

    Args:
        csv_path (str): Path to CSV file with mutations
        model_type (str): Type of model to use ('esmc_300M', 'esmc_600M', or 'esm3_open')
        model_path (str, optional): Path to local model checkpoint (if None, downloads from HuggingFace)
        pdb_path (str, optional): Path to PDB file (required for esm3_open with use_structure=True)
        sequence_file (str, optional): Path to file containing protein sequence
        sequence (str, optional): Direct protein sequence string
        chain_id (str, optional): Chain ID to use from PDB (for esm3_open with use_structure=True)
        use_structure (bool): Whether to use structure information (for esm3_open)
        pdb_range (str, optional): Range of residues in the sequence covered by the PDB (format: "start-end", 1-indexed)
        output_path (str, optional): Path to save output CSV

    Returns:
        float: Spearman correlation coefficient
    """
    assert (model_type == 'esm3_open' and use_structure) if scoring_method == 'joint_marginal' else True
    
    # Load CSV file
    if os.path.exists(output_path):
        csv_path = output_path
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} mutations from CSV file")

    # Extract mutation strings from CSV
    mutations = df['mutant'].tolist()

    # Initialize model
    print(f"Loading {model_type.upper()} model")
    if model_type in ["esmc_300M", "esmc_600M"]:
        if model_path:
            print(f"Loading {model_type} from local path: {model_path}")
            model = ESMC.from_pretrained(model_path).to("cuda")
        else:
            model_name = "esmc_300m" if model_type == "esmc_300M" else "esmc_600m"
            print(f"Downloading {model_type} from HuggingFace")
            model = ESMC.from_pretrained(model_name).to("cuda")
        
        # Get protein sequence
        if sequence is None:
            if sequence_file is None:
                raise ValueError(f"For {model_type} model, either sequence or sequence_file must be provided")
            
            with open(sequence_file, 'r') as f:
                lines = f.readlines()
                # Skip header line if it starts with >
                sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
        
        # Score mutations
        print(f"Scoring mutations with {model_type}")
        mutation_scores = score_mutations(
            sequence,
            mutations,
            model,
            model_type=model_type,
            model_path=model_path,
            window_size=1024
        )
        
        # Add scores to dataframe
        score_column = f'{model_type}_score'
        
    elif model_type == "esm3_open":
        if model_path:
            print(f"Loading ESM3_open from local path: {model_path}")
            model = ESM3.from_pretrained(model_path).to("cuda")
        else:
            print("Downloading ESM3_open from HuggingFace")
            model = ESM3.from_pretrained("esm3-open").to("cuda")
        
        if use_structure:
            if pdb_path is None:
                raise ValueError("For ESM3_open model with use_structure=True, pdb_path must be provided")
                
            print(f"Scoring mutations with ESM3_open.{scoring_method} using structure")
            mutation_scores = score_mutations_with_pdb(
                pdb_path,
                mutations,
                chain_id,
                model,
                model_type=model_type,
                model_path=model_path,
                use_structure=True,
                window_size=1024,
                pdb_range=pdb_range,
                scoring_method=scoring_method,
                devices=devices,
                batch_size=batch_size
            )

        else:
            # For ESM3 without structure, use sequence directly 
            if sequence is None:
                if sequence_file is None:
                    raise ValueError("For ESM3_open model without structure, either sequence or sequence_file must be provided")
                
                with open(sequence_file, 'r') as f:
                    lines = f.readlines()
                    # Skip header line if it starts with >
                    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
            
            # Score mutations with sequence only
            print("Scoring mutations with ESM3_open using sequence only")
            mutation_scores = score_mutations(
                sequence,
                mutations,
                model,
                model_type=model_type,
                model_path=model_path,
                window_size=1024,
                devices=devices
            )
        
        # Add scores to dataframe
        # score_column = 'ESM3_open_score'
        score_column = f'ESM3.{scoring_method}'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Add scores to dataframe
    df[score_column] = df['mutant'].map(lambda x: mutation_scores.get(x, np.nan))

    # Calculate Spearman correlation
    valid_data = df.dropna(subset=[score_column, 'DMS_score'])
    if len(valid_data) > 0:
        correlation, p_value = spearmanr(valid_data['DMS_score'], valid_data[score_column])
        print(f"\nSpearman correlation: {correlation:.4f} (p-value: {p_value:.4e})")
        print(f"Based on {len(valid_data)} valid mutations (out of {len(df)} total)")
    else:
        correlation = np.nan
        print("No valid mutations for correlation calculation")
        raise

    # Save results to CSV
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return correlation


def process_assays_from_file(input_list_csv, base_dms_dir, pdb_dir, output_dir, model_type="esmc_300M", model_path=None, use_structure=False, dms_index=-1, scoring_method='default', devices=[0], batch_size=1):
    """
    Process multiple assays from a CSV file with DMS_id column.

    Args:
        input_list_csv (str): Path to CSV file with list of assays (containing DMS_id and target_seq columns)
        base_dms_dir (str): Base directory containing DMS CSV files
        pdb_dir (str): Base directory containing PDB files (only needed for esm3_open with use_structure=True)
        output_dir (str): Directory to save output files
        model_type (str): Type of model to use ('esmc_300M', 'esmc_600M', or 'esm3_open')
        model_path (str, optional): Path to local model checkpoint (if None, downloads from HuggingFace)
        use_structure (bool): Whether to use structure information (for esm3_open)
        dms_index (int): Index of DMS to score. If -1, score all DMS assays

    Returns:
        dict: Dictionary of assay IDs and their correlation values
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV file with assay list
    assay_list_df = pd.read_csv(input_list_csv)

    if 'DMS_id' not in assay_list_df.columns:
        raise ValueError("Input CSV must contain a 'DMS_id' column")
        
    if 'target_seq' not in assay_list_df.columns:
        raise ValueError("Input CSV must contain a 'target_seq' column with protein sequences")

    if use_structure and 'pdb_file' not in assay_list_df.columns:
        raise ValueError("Input CSV must contain a 'pdb_file' column if we use structures for scoring")
    
    has_pdb_range = 'pdb_range' in assay_list_df.columns

    results = {}

    # If dms_index is specified and valid, process only that assay
    if dms_index != -1:
        try:
            dms_index = int(dms_index)
            if 0 <= dms_index < len(assay_list_df):
                assay_list_df = assay_list_df.iloc[[dms_index]]
                print(f"Processing only DMS at index {dms_index}: {assay_list_df.iloc[0]['DMS_id']}")
            else:
                print(f"Warning: DMS_index {dms_index} out of range (0-{len(assay_list_df)-1}), processing all assays")
        except ValueError:
            print(f"Warning: Invalid DMS_index '{dms_index}', processing all assays")

    # Process each assay
    for idx, row in assay_list_df.iterrows():
        assay = row['DMS_id']
        target_sequence = row['target_seq']
        pdb_file = row['pdb_file'] if use_structure else None
        pdb_range = row['pdb_range'] if has_pdb_range and use_structure else None
        if pdb_range:
            print(f"PDB range specified: {pdb_range}")
        
        print(f"\n=== Processing assay {assay} ({idx+1}/{len(assay_list_df)}) ===")

        # Construct paths
        input_csv = os.path.join(base_dms_dir, f"{assay}.csv")
        output_csv = os.path.join(output_dir, f"{assay}.csv")

        # Check if input CSV exists
        if not os.path.exists(input_csv):
            print(f"Error: Input CSV file {input_csv} not found, skipping")
            continue

        # Process based on model type
        # try:
        if model_type in ["esmc_300M", "esmc_600M"]:
            # For ESM-C, use sequence from reference file
            # Process the assay with sequence from reference file
            correlation = process_csv_and_score_mutations(
                input_csv,
                model_type=model_type,
                model_path=model_path,
                sequence=target_sequence,
                output_path=output_csv,
                devices=devices,
                batch_size=batch_size
            )
        else:  # esm3_open
            if use_structure:
                # For ESM3 with structure, use PDB file
                pdb_path = os.path.join(pdb_dir, pdb_file)
                
                if not os.path.exists(pdb_path):
                    print(f"Error: PDB file {pdb_path} not found, skipping")
                    continue
                    
                # Process the assay with structure
                correlation = process_csv_and_score_mutations(
                    input_csv,
                    model_type=model_type,
                    model_path=model_path,
                    pdb_path=pdb_path,
                    chain_id="A",  # Default chain ID
                    use_structure=True,
                    pdb_range=pdb_range,
                    output_path=output_csv,
                    scoring_method=scoring_method,
                    batch_size=batch_size
                )
            else:
                # For ESM3 without structure, use sequence from reference file
                correlation = process_csv_and_score_mutations(
                    input_csv,
                    model_type=model_type,
                    model_path=model_path,
                    sequence=target_sequence,
                    use_structure=False,
                    output_path=output_csv,
                    batch_size=batch_size
                )

        results[assay] = correlation
        # except Exception as e:
        #     print(f"Error processing {assay}: {str(e)}")
        #     results[assay] = np.nan

    # Create summary file with correlations
    summary_df = pd.DataFrame(
        {'assay': list(results.keys()), 'correlation': list(results.values())}
    )
    summary_file_path = os.path.join(output_dir, f"correlation_summary_{model_type}.csv")
    if os.path.exists(summary_file_path):
        # Append without including headers again
        summary_df.to_csv(summary_file_path, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        summary_df.to_csv(summary_file_path, index=False)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process DMS assays with ESM model scoring")
    parser.add_argument("--model_type", choices=["esmc_300M", "esmc_600M", "esm3_open"], default="esmc_300M",
                      help="Model type to use for scoring (esmc_300M, esmc_600M, or esm3_open)")
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to local model checkpoint (if not provided, downloads from HuggingFace)")
    parser.add_argument("--reference_csv", required=True, 
                      help="CSV file with DMS_id and target_seq columns")
    parser.add_argument("--dms_dir", required=True,
                      help="Directory containing DMS CSV files")
    parser.add_argument("--pdb_dir", required=False, default=None,
                      help="Directory containing PDB files (only needed for esm3_open with use_structure=True)")
    parser.add_argument("--output_dir", required=True,
                      help="Directory to save output files")
    parser.add_argument("--use_structure", action="store_true", default=True,
                      help="Whether to use structure information (for esm3_open only)")
    parser.add_argument("--DMS_index", required=False, default=-1,
                      help="Index of DMS to score. If not provided, score all DMS assays")

    parser.add_argument("--scoring_method", type=str, default="default", 
                       help="Scoring method to use (default, joint)")
    parser.add_argument("--devices", type=int, default=0, nargs="+")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    
    # Verify pdb_dir is provided if using structure
    if args.model_type == "esm3_open" and args.use_structure and args.pdb_dir is None:
        parser.error("--pdb_dir is required when using esm3_open with structure enabled")

    # Process all assays from the input list
    results = process_assays_from_file(
        args.reference_csv,
        args.dms_dir,
        args.pdb_dir,
        args.output_dir,
        args.model_type,
        args.model_path,
        args.use_structure,
        args.DMS_index,
        args.scoring_method,
        args.devices,
        batch_size = args.batch_size
    )

    # Print summary of results
    print("\n=== Summary of Results ===")
    for assay, correlation in results.items():
        if not np.isnan(correlation):
            print(f"{assay}: Spearman correlation = {correlation:.4f}")
        else:
            print(f"{assay}: Failed to calculate correlation")