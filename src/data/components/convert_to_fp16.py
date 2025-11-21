import torch
import os
from pathlib import Path

def convert_shard_to_fp16(file_path, output_dir):
    """Loads a shard, converts to fp16, and saves it."""
    try:
        # Load the data
        data = torch.load(file_path)
        
        converted = False
        
        if isinstance(data, dict):
            if 'embeddings' in data and isinstance(data['embeddings'], list):
                new_embeddings = []
                for emb in data['embeddings']:
                    if isinstance(emb, torch.Tensor):
                        new_embeddings.append(emb.half())
                    else:
                        new_embeddings.append(emb)
                data['embeddings'] = new_embeddings
                converted = True
            else:
                print(f"Skipping {file_path}: Dict does not contain 'embeddings' list.")
        elif isinstance(data, torch.Tensor):
            data = data.half()
            converted = True
        else:
            print(f"Skipping {file_path}: Unknown data type {type(data)}")
            return

        if not converted:
             print(f"Skipping {file_path}: No conversion performed.")
             return

        # Construct new filename
        # Expected format: esm2_token_embeddings_shard_XXX.pt
        # New format: esm2_token_embeddings_shard_fp16_XXX.pt
        filename = file_path.name
        parts = filename.split('_')
        # parts: ['esm2', 'token', 'embeddings', 'shard', 'XXX.pt']
        
        if len(parts) < 2:
             print(f"Skipping {file_path}: Unexpected filename format.")
             return

        # Insert 'fp16' before the last part (the number)
        new_parts = parts[:-1] + ['fp16', parts[-1]]
        new_filename = "_".join(new_parts)
        
        new_file_path = output_dir / new_filename
        
        # Save
        torch.save(data, new_file_path)
        print(f"Saved {new_file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Define the directory containing the shards
    # Assuming the script is run from the project root
    data_dir = Path("data/esm2_token_embeddings_sharded")
    output_dir = Path("data/esm2_token_embeddings_sharded_fp16")
    
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist.")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all .pt files that match the pattern and don't have 'fp16' in the name
    files = sorted([
        f for f in data_dir.glob("*.pt") 
        if "fp16" not in f.name and "esm2_token_embeddings_shard" in f.name
    ])
    
    print(f"Found {len(files)} files to convert.")
    
    for i, file_path in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {file_path.name}")
        convert_shard_to_fp16(file_path, output_dir)

if __name__ == "__main__":
    main()
