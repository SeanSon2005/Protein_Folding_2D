import sys
import os
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(".."))

from src.data.constants import INPUT_ALPHABET, CANONICAL_AA
from src.models.protein_module import ProteinLitModule

CHECKPOINT_PATH = "logs/train/runs/2025-11-19_16-34-55/checkpoints/last.ckpt"
DATA_PATH = "data/test/test.tsv"
VOCAB_PATH = "data/ps4_data.csv"
OUTPUT_FILE = "predictions/predictions_1.csv"

# Build vocabs
input_vocab = {c: i + 1 for i, c in enumerate(INPUT_ALPHABET)}
input_vocab["<PAD>"] = 0

# Get target vocab from training data
df = pd.read_csv(VOCAB_PATH)
target_chars = set()
for dssp in df['dssp8']:
    target_chars.update(dssp)
target_vocab = {c: i + 1 for i, c in enumerate(sorted(list(target_chars)))}
target_vocab["<PAD>"] = 0
inv_target_vocab = {v: k for k, v in target_vocab.items()}

print(f"Input Vocab: {input_vocab}")
print(f"Target Vocab: {target_vocab}")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinLitModule.load_from_checkpoint(CHECKPOINT_PATH, map_location=device)
model.to(device)
model.eval()
model.freeze()
print("Model loaded successfully")

# Load Test Data
AA_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "ASX": "X",
    "AYA": "X",
    "CAF": "X",
    "CGU": "E",
    "CME": "C",
    "CSD": "C",
    "CSO": "C",
    "CSS": "C",
    "CYG": "C",
    "CYS": "C",
    "DAH": "X",
    "FC0": "X",
    "FME": "M",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "KCX": "K",
    "KPI": "K",
    "LED": "X",
    "LEU": "L",
    "LLP": "K",
    "LVN": "X",
    "LYS": "K",
    "MET": "M",
    "MLE": "L",
    "MLY": "K",
    "MLZ": "K",
    "MSE": "M",
    "MSO": "M",
    "OCS": "C",
    "PCA": "E",
    "PHE": "F",
    "PRO": "P",
    "PYL": "K",
    "RGL": "R",
    "SCH": "C",
    "SEC": "C",
    "SEP": "S",
    "SER": "S",
    "SNN": "X",
    "THR": "T",
    "TRO": "W",
    "TRP": "W",
    "TSY": "Y",
    "TYR": "Y",
    "UNK": "X",
    "VAL": "V",
    "WLU": "W",
    "WPA": "W",
    "WRP": "W",
    "WVL": "W",
}

unique_ids = set()
test_rows = []
residue_map = defaultdict(dict)
max_indices = defaultdict(int)
with open(DATA_PATH) as f:
    header = next(f).strip()
    for line in f:
        line = line.strip()
        if line:
            test_rows.append(line)
            seq_id, res_name, res_idx_str = line.split('_')
            res_idx = int(res_idx_str)
            unique_ids.add(seq_id)
            residue_map[seq_id][res_idx] = res_name
            if res_idx > max_indices[seq_id]:
                max_indices[seq_id] = res_idx

print(f"Found {len(unique_ids)} unique sequences in test set")

test_seqs = {}
for seq_id in unique_ids:
    length = max_indices[seq_id]
    residues = []
    seq_lookup = residue_map[seq_id]
    for idx in range(1, length + 1):
        aa_three = seq_lookup.get(idx, "UNK")
        residues.append(AA_THREE_TO_ONE.get(aa_three, "X"))
    test_seqs[seq_id] = "".join(residues)

print(f"Constructed {len(test_seqs)} sequences directly from test.tsv")

# Inference
predictions = {}

print("Running inference...")
for seq_id, seq in tqdm(test_seqs.items()):
    # Tokenize
    normalized_seq = [c if c in CANONICAL_AA else "X" for c in seq.upper()]
    indices = [input_vocab[c] for c in normalized_seq]
    tensor = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        logits = model(tensor) # [1, L, C]
        preds = torch.argmax(logits, dim=2).squeeze(0) # [L]
        
    # Decode
    pred_chars = [inv_target_vocab[idx.item()] for idx in preds]
    predictions[seq_id] = "".join(pred_chars)

print("Inference complete")

# Generate Submission
rows = []
for row in test_rows:
    parts = row.split('_')
    seq_id = parts[0]
    res_idx = int(parts[-1])

    if seq_id in predictions:
        pred_seq = predictions[seq_id]
        if res_idx <= len(pred_seq):
            pred_char = pred_seq[res_idx - 1]
        else:
            pred_char = "."
    else:
        pred_char = "."
    if pred_char == "C":
        pred_char = "."
    rows.append({"id": row, "secondary_structure": pred_char})

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False, sep='\t')

print(f"Saved {OUTPUT_FILE}")