import torch
import glob
import os
from pathlib import Path
from tqdm import tqdm
import gc
import shutil
import math

def reshard_dataset(
    source_dir: str,
    target_dir: str,
    temp_dir: str = "data/temp_reshard",
    target_shard_size_limit: int = 2 * 1024 * 1024 * 1024  # 2GB
):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    temp_path = Path(temp_dir)
    
    # Cleanup and setup
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"1. Scanning metadata in {source_path}...")
    shard_files = sorted(list(source_path.glob("*.pt")))
    all_metadata = []
    
    for shard_file in tqdm(shard_files, desc="Scanning"):
        try:
            data = torch.load(shard_file, map_location="cpu")
            lengths = data["lengths"]
            if torch.is_tensor(lengths):
                lengths = lengths.tolist()
            chain_ids = data["chain_ids"]
            
            # We only need length to determine the bucket
            for i, (chain_id, length) in enumerate(zip(chain_ids, lengths)):
                all_metadata.append({
                    "length": int(length),
                    "chain_id": chain_id,
                    "shard_path": str(shard_file),
                    "index": i
                })
            del data
            gc.collect()
        except Exception as e:
            print(f"Error reading {shard_file}: {e}")

    total_samples = len(all_metadata)
    print(f"Found {total_samples} samples.")
    
    # 2. Define Buckets (Target Shards)
    print("2. Defining buckets...")
    all_metadata.sort(key=lambda x: x["length"])
    
    # We need to estimate how many samples fit in 2GB.
    # We don't know exact sizes, but we can assume roughly proportional to length?
    # Or just use a fixed number of samples per shard?
    # Let's assume average protein is ~400 residues * 1280 dim * 2 bytes (fp16) ~= 1MB.
    # 2GB ~= 2000 samples.
    # Let's try to balance by count for now, or refine if we had size info.
    # The previous script used actual size.
    # Let's use a simple heuristic: assume average size.
    # Or better: we can just partition the SORTED metadata into N chunks.
    # But we want chunks to be roughly 2GB.
    # Let's assume we want ~20 shards (based on input size).
    
    # Let's just divide into N buckets where N = total_size / 2GB.
    # We don't know total size exactly, but input is ~40GB. So ~20 buckets.
    num_buckets = 25 # Safety margin
    samples_per_bucket = math.ceil(total_samples / num_buckets)
    
    buckets = []
    for i in range(0, total_samples, samples_per_bucket):
        bucket_meta = all_metadata[i : i + samples_per_bucket]
        if not bucket_meta:
            continue
        # Range of lengths in this bucket
        min_len = bucket_meta[0]["length"]
        max_len = bucket_meta[-1]["length"]
        buckets.append({
            "id": len(buckets),
            "min_len": min_len,
            "max_len": max_len,
            "count": len(bucket_meta),
            "file_parts": []
        })
    
    print(f"Created {len(buckets)} buckets based on length ranges.")
    
    # Map length to bucket index
    # Since buckets are sorted by length ranges, we can use bisect or just linear search (small number of buckets)
    # But ranges might overlap if many samples have same length?
    # Actually, we just need to know which bucket a sample belongs to.
    # We can build a lookup or just use the sorted metadata to assign bucket IDs?
    # Wait, we are iterating SOURCE shards, so we get samples in random order.
    # We need a quick way to know: "Sample with length L and chain_id C goes to bucket B".
    # Since (chain_id) is unique, we can map chain_id -> bucket_id.
    
    chain_to_bucket = {m["chain_id"]: b_idx for b_idx, bucket in enumerate(buckets) for m in all_metadata[bucket["id"] * samples_per_bucket : (bucket["id"] + 1) * samples_per_bucket]}
    
    # Free metadata to save RAM
    del all_metadata
    gc.collect()
    
    # 3. Partition Phase
    print("3. Partitioning data into temporary files...")
    
    # Buffers for each bucket: list of (embedding, chain_id, length)
    bucket_buffers = [[] for _ in range(len(buckets))]
    BUFFER_SIZE_LIMIT = 50 * 1024 * 1024  # 50MB per bucket buffer
    
    for shard_file in tqdm(shard_files, desc="Processing source shards"):
        try:
            data = torch.load(shard_file, map_location="cpu")
            embeddings = data["embeddings"]
            chain_ids = data["chain_ids"]
            lengths = data["lengths"]
            if torch.is_tensor(lengths):
                lengths = lengths.tolist()
                
            for i in range(len(chain_ids)):
                chain_id = chain_ids[i]
                if chain_id not in chain_to_bucket:
                    continue # Should not happen
                
                b_idx = chain_to_bucket[chain_id]
                emb = embeddings[i]
                length = lengths[i]
                
                bucket_buffers[b_idx].append((emb, chain_id, length))
                
                # Check buffer size (approx)
                # simple check: count items? 
                # Let's flush every 100 items to be safe and simple
                if len(bucket_buffers[b_idx]) >= 100:
                    # Flush
                    part_file = temp_path / f"bucket_{b_idx:03d}_part_{len(buckets[b_idx]['file_parts']):04d}.pt"
                    torch.save(bucket_buffers[b_idx], part_file)
                    buckets[b_idx]['file_parts'].append(part_file)
                    bucket_buffers[b_idx] = []
            
            del data
            del embeddings
            del chain_ids
            del lengths
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {shard_file}: {e}")

    # Flush remaining buffers
    for b_idx in range(len(buckets)):
        if bucket_buffers[b_idx]:
            part_file = temp_path / f"bucket_{b_idx:03d}_part_{len(buckets[b_idx]['file_parts']):04d}.pt"
            torch.save(bucket_buffers[b_idx], part_file)
            buckets[b_idx]['file_parts'].append(part_file)
            bucket_buffers[b_idx] = []

    # 4. Merge Phase
    print("4. Merging and sorting final shards...")
    
    for b_idx, bucket in enumerate(tqdm(buckets, desc="Finalizing buckets")):
        all_samples = []
        for part_file in bucket['file_parts']:
            part_data = torch.load(part_file, map_location="cpu")
            all_samples.extend(part_data)
            # Delete part file to free space
            os.remove(part_file)
            
        # Sort by length
        all_samples.sort(key=lambda x: x[2]) # (emb, chain_id, length)
        
        # Unzip
        if not all_samples:
            continue
            
        final_embeddings = [x[0] for x in all_samples]
        final_chain_ids = [x[1] for x in all_samples]
        final_lengths = [x[2] for x in all_samples]
        
        output_file = target_path / f"esm2_token_embeddings_shard_fp16_sorted_{b_idx:03d}.pt"
        torch.save({
            "embeddings": final_embeddings,
            "chain_ids": final_chain_ids,
            "lengths": torch.tensor(final_lengths, dtype=torch.long)
        }, output_file)
        
        del all_samples
        del final_embeddings
        del final_chain_ids
        del final_lengths
        gc.collect()

    # Cleanup
    shutil.rmtree(temp_path)
    print("Done!")

if __name__ == "__main__":
    reshard_dataset(
        source_dir="data/esm2_token_embeddings_sharded_fp16",
        target_dir="data/esm2_token_embeddings_sharded_fp16_sorted"
    )
