import sys
import os
import csv
from collections import defaultdict
import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.models.protein_module import ProteinLitModule

def generate_submission():
    print("Starting submission generation with fixed embeddings...")
    
    # Constants
    CHECKPOINT_PATH = "logs/train/runs/2025-11-21_01-52-16/checkpoints/last.ckpt"
    EMBEDDINGS_PATH = "data/esm2_embeddings_test_fixed.pt"
    TEST_FILE = "data/test/test.tsv"
    OUTPUT_FILE = "predictions/predictions_2.csv"
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
        model = ProteinLitModule.load_from_checkpoint(CHECKPOINT_PATH, map_location=device)
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
    embedding_map = {cid: emb for cid, emb in zip(chain_ids, embeddings)}
    print(f"Loaded {len(embedding_map)} embeddings.")

    # 4. Run Inference
    print("Running inference...")
    predictions = {} # seq_id -> string of predictions
    
    for seq_id, emb in tqdm(embedding_map.items()):
        # emb is [L, D]
        tensor = emb.unsqueeze(0).to(device) # [1, L, D]
        with torch.no_grad():
            logits = model(tensor) # [1, L, C]
            preds = torch.argmax(logits, dim=2).squeeze(0) # [L]
        
        pred_chars = [inv_target_vocab[idx.item()] for idx in preds]
        predictions[seq_id] = "".join(pred_chars)

    # 5. Generate Submission
    # Since embeddings were generated with correct spacing (1-based index maps to index-1),
    # we can now just look up directly.
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
