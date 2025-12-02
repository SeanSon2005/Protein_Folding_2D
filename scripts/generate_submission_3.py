import sys
import os
import csv
from collections import defaultdict
import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.models.protein_module import ProteinResCRFLitModule

def generate_submission():
    print("Starting submission generation with fixed embeddings...")
    
    # Constants
    CHECKPOINT_PATH = "logs/train/runs/2025-12-01_22-46-56/checkpoints/last.ckpt"
    EMBEDDINGS_PATH = "data/esm2_embeddings_test_fixed.pt"
    TEST_FILE = "data/test/test.tsv"
    OUTPUT_FILE = "predictions/predictions_3.csv"
    DATA_FILE = "data/ps4_data.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build Target Vocab
    print("Building target vocab...")
    df = pd.read_csv(DATA_FILE)
    target_chars = set()
    for dssp in df['dssp8']:
        target_chars.update(dssp)
    target_vocab = {c: i + 1 for i, c in enumerate(sorted(list(target_chars)))}
    target_vocab["<PAD>"] = 0
    inv_target_vocab = {v: k for k, v in target_vocab.items()}
    print(f"Target Vocab: {target_vocab}")

    # 2. Load Model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    try:
        model = ProteinResCRFLitModule.load_from_checkpoint(CHECKPOINT_PATH, map_location=device)
        model.to(device)
        model.eval()
        model.freeze()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Load Embeddings
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    data = torch.load(EMBEDDINGS_PATH)
    chain_ids = data["chain_ids"]
    embeddings = data["embeddings"]
    lengths = data["lengths"]
    embedding_map = {cid: (emb, length) for cid, emb, length in zip(chain_ids, embeddings, lengths)}
    print(f"Loaded {len(embedding_map)} embeddings.")

    # 4. Load Residue Ids from test file
    print("Loading residue IDs from test file...")
    RESIDUE_VOCAB = {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
        "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14,
        "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21,
        "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26,
    }
    
    # 3-letter to 1-letter amino acid mapping
    AA_3TO1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "SEC": "U", "PYL": "O",  # Rare amino acids
    }
    
    # Build position -> amino acid mapping from test file
    chain_residues = defaultdict(dict)  # chain_id -> {pos: 1-letter AA}
    with open(TEST_FILE, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            if not row:
                continue
            line_id = row[0]
            parts = line_id.split('_')
            seq_id = parts[0]
            aa_3letter = parts[1]
            pos = int(parts[2])
            aa_1letter = AA_3TO1.get(aa_3letter, "X")
            chain_residues[seq_id][pos] = aa_1letter
    
    # Build full-length residue_ids for each chain (matching embedding length)
    for chain_id in data['chain_ids']:
        if chain_id not in chain_residues:
            print(f"Warning: Chain ID {chain_id} not found in test file.")
            continue
        
        if chain_id not in embedding_map:
            print(f"Warning: Chain ID {chain_id} not found in embeddings.")
            continue
        
        emb, emb_length = embedding_map[chain_id]
        pos_to_aa = chain_residues[chain_id]
        
        residue_ids = []
        for i in range(emb_length):
            pos = i + 1  # 1-based position
            if pos in pos_to_aa:
                aa = pos_to_aa[pos]
            else:
                aa = "X"  # Unknown/missing position
            residue_ids.append(RESIDUE_VOCAB.get(aa, 0))
        
        embedding_map[chain_id] = (emb, torch.tensor(residue_ids, dtype=torch.long))


    # 5. Run Inference
    print("Running inference...")
    predictions = {} # seq_id -> string of predictions
    
    for seq_id, (emb, residue_ids) in tqdm(embedding_map.items()):
        # emb is [L, D]
        tensor = emb.unsqueeze(0).to(device) # [1, L, D]
        residue_ids = residue_ids.unsqueeze(0).to(device) # [1, L]
        with torch.no_grad():
            logits = model(tensor, residue_ids) # [1, L, C]
            preds = torch.argmax(logits, dim=2).squeeze(0) # [L]
        
        pred_chars = [inv_target_vocab[idx.item()] for idx in preds]
        predictions[seq_id] = "".join(pred_chars)

    # 6. Generate Submission
    print("Generating submission...")
    
    rows = []
    with open(TEST_FILE, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            if not row: continue
            line_id = row[0]
            parts = line_id.split('_')
            seq_id = parts[0]
            pos = int(parts[2])
            
            pred_char = "."
            if seq_id in predictions:
                pred_seq = predictions[seq_id]
                # pos is 1-based. Index is pos-1.
                idx = pos - 1
                
                if idx < len(pred_seq):
                    pred_char = pred_seq[idx]
                else:
                    # Should not happen if max_pos logic was correct
                    pass
            
            if pred_char == "C":
                pred_char = "."
                
            rows.append({"id": line_id, "secondary_structure": pred_char})

    df_submission = pd.DataFrame(rows)
    df_submission.to_csv(OUTPUT_FILE, index=False, sep='\t')
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_submission()
