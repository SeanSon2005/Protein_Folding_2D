import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BitsAndBytesConfig
import csv
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

# Constants
DATA_FILE = "data/test/test.tsv"
OUTPUT_FILE = "data/esm2_embeddings_test_fixed.pt"
MODEL_NAME = "facebook/esm2_t48_15B_UR50D"

def parse_test_tsv(file_path):
    """
    Parses test.tsv and reconstructs FULL sequences with gaps filled by 'X'.
    """
    sequences = defaultdict(list)
    
    # 3-letter to 1-letter map
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    print(f"Reading {file_path}...")
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader) # Skip header "id"
        
        for row in reader:
            if not row: continue
            # Format: 3JRN_LYS_8
            parts = row[0].split('_')
            if len(parts) != 3:
                continue
                
            pdb_id = parts[0]
            aa_3 = parts[1]
            pos = int(parts[2]) # 1-based index
            
            aa_1 = aa_map.get(aa_3, 'X')
            sequences[pdb_id].append((pos, aa_1))
            
    # Reconstruct sequences with correct spacing
    final_data = []
    print(f"Reconstructing sequences for {len(sequences)} proteins...")
    
    for pdb_id, residue_list in sequences.items():
        if not residue_list: continue
        
        # Find max position to determine sequence length
        max_pos = max(pos for pos, _ in residue_list)
        
        # Initialize sequence with 'X' (or <UNK>)
        # We will use 'X' as the placeholder for unknown/gap residues
        seq_array = ['X'] * max_pos
        
        for pos, aa in residue_list:
            # pos is 1-based, so index is pos-1
            if 1 <= pos <= max_pos:
                seq_array[pos-1] = aa
        
        seq = "".join(seq_array)
        final_data.append({"chain_id": pdb_id, "input": seq})
        
    return final_data

class ProteinTestDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        # Tokenize the sequence
        # We use the tokenizer directly to handle special tokens if needed, 
        # but for ESM we usually just map characters. 
        # Let's use the tokenizer to be safe and consistent with the model.
        # However, we need to be careful about special tokens (CLS, EOS).
        # The tokenizer usually adds them.
        
        # Note: ESM tokenizer treats 'X' as <UNK> usually.
        return row['input'], row['chain_id']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Load Model with 4-bit Quantization
    print(f"Loading model {MODEL_NAME} with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # 3. Prepare Data
    data = parse_test_tsv(DATA_FILE)
    dataset = ProteinTestDataset(data, tokenizer)
    
    # 4. Generate Embeddings
    all_chain_ids = []
    all_lengths = []
    all_embeddings = []
    
    print("Starting embedding generation...")
    # Process one by one to manage memory safely
    with torch.no_grad():
        for seq, chain_id in tqdm(dataset):
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Last layer hidden states: (1, seq_len_with_specials, hidden_dim)
            hidden = outputs.hidden_states[-1]
            
            # Remove CLS (first) and EOS (last) tokens
            # input_ids has shape [1, L+2]
            seq_len = input_ids.shape[1] - 2
            
            # Extract embeddings for the sequence only
            # hidden: [1, L+2, D] -> slice [1, 1:L+1, :]
            seq_hidden = hidden[0, 1:seq_len+1, :]
            
            # Move to CPU and float32
            seq_hidden_cpu = seq_hidden.detach().cpu().to(torch.float32)
            
            all_chain_ids.append(chain_id)
            all_lengths.append(seq_len)
            all_embeddings.append(seq_hidden_cpu)

    # 5. Save
    print(f"Saving {len(all_chain_ids)} embeddings to {OUTPUT_FILE}...")
    obj = {
        "chain_ids": all_chain_ids,
        "lengths": torch.tensor(all_lengths, dtype=torch.int32),
        "embeddings": all_embeddings,
    }
    torch.save(obj, OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    main()
