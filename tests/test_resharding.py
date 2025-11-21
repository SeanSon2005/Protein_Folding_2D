import torch
from pathlib import Path
import random
import sys

def verify_resharding(original_dir, new_dir):
    print(f"Verifying {new_dir} against {original_dir}...")
    
    orig_path = Path(original_dir)
    new_path = Path(new_dir)
    
    # 1. Count samples
    def count_samples(path):
        files = sorted(list(path.glob("*.pt")))
        count = 0
        sample_map = {} # chain_id -> (shard_path, index)
        for f in files:
            data = torch.load(f, map_location="cpu")
            chain_ids = data["chain_ids"]
            for i, cid in enumerate(chain_ids):
                sample_map[cid] = (str(f), i)
            count += len(chain_ids)
        return count, sample_map

    print("Counting original samples...")
    orig_count, orig_map = count_samples(orig_path)
    print(f"Original samples: {orig_count}")
    
    print("Counting new samples...")
    new_count, new_map = count_samples(new_path)
    print(f"New samples: {new_count}")
    
    if orig_count != new_count:
        print(f"ERROR: Sample count mismatch! {orig_count} vs {new_count}")
        sys.exit(1)
        
    # 2. Verify content for random samples
    print("Verifying content for 10 random samples...")
    chain_ids = list(orig_map.keys())
    random.shuffle(chain_ids)
    
    for cid in chain_ids[:10]:
        orig_shard, orig_idx = orig_map[cid]
        new_shard, new_idx = new_map[cid]
        
        orig_data = torch.load(orig_shard, map_location="cpu")
        orig_emb = orig_data["embeddings"][orig_idx]
        
        new_data = torch.load(new_shard, map_location="cpu")
        new_emb = new_data["embeddings"][new_idx]
        
        if not torch.equal(orig_emb, new_emb):
            print(f"ERROR: Embedding mismatch for chain {cid}")
            sys.exit(1)
            
    print("SUCCESS: Verification passed!")

if __name__ == "__main__":
    verify_resharding(
        "data/esm2_token_embeddings_sharded_fp16",
        "data/esm2_token_embeddings_sharded_fp16_sorted"
    )
