import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
from tqdm import tqdm

_ALIGNER: Optional[PairwiseAligner] = None


def _get_aligner() -> PairwiseAligner:
    """Lazily create and reuse a global aligner per process."""
    global _ALIGNER
    if _ALIGNER is None:
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -1.0
        _ALIGNER = aligner
    return _ALIGNER


def _get_mmcif_dict(cif_path: str) -> MMCIF2Dict:
    # No caching to avoid memory growth across many files
    return MMCIF2Dict(cif_path)


def _get_structure(pdb_id: str, cif_path: str):
    # Fresh parser per call to avoid retaining parsed structures in memory
    parser = MMCIFParser(QUIET=True, auth_chains=False)
    return parser.get_structure(pdb_id, cif_path)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Cα–Cα contact maps from CIF files (sequence-aligned)."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/ps4_data.csv",
        help="Path to the CSV file containing chain_id and input columns",
    )
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default="data/mmcif",
        help="Directory containing .cif files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/contact_maps",
        help="Directory to save contact map .pt files",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff in Angstroms for defining contacts (default: 8.0)",
    )
    parser.add_argument(
        "--min_seq_sep",
        type=int,
        default=4,
        help="Minimum sequence separation |i-j| to consider a contact (default: 4)",
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of PS4 sequence positions that must map to "
            "structure residues; below this, the entry is skipped (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes (<=0 to use all cores, 1 for single-core).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing .pt files instead of skipping them.",
    )
    return parser.parse_args()


def get_chain_residues_with_ca(structure, chain_id: str):
    """Return a list of residues for the given chain that are:
    - standard residues (hetfield == ' ')
    - have a Cα atom
    in coordinate order.
    """
    # Use first model only
    chain = None
    for model in structure:
        if chain is not None:
            break
        for c in model:
            if c.id == chain_id:
                chain = c
                break

    if chain is None:
        raise ValueError(f"Chain '{chain_id}' not found in structure {structure.id}")

    residues = []
    for res in chain:
        hetfield, resseq, icode = res.id
        if hetfield != " ":
            continue
        if "CA" not in res:
            continue
        residues.append(res)

    if not residues:
        raise ValueError(
            f"No standard residues with Cα found in chain '{chain_id}' of {structure.id}"
        )

    return residues


def residues_to_sequence(residues) -> str:
    """Convert a list of Bio.PDB Residue objects to a 1-letter AA sequence.
    Unknown / modified residues are mapped to 'X'.
    """
    three_to_one_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
        "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L", "LYS": "K",
        "MET": "M",
        "PHE": "F", "PRO": "P",
        "SER": "S",
        "THR": "T", "TRP": "W", "TYR": "Y",
        "VAL": "V",
    }

    letters = []
    for res in residues:
        resname = res.get_resname().upper()  # e.g. 'ALA', 'MSE', ...
        aa = three_to_one_map.get(resname, "X")
        letters.append(aa)
    return "".join(letters)



def align_ps4_to_structure(
    ps4_seq: str,
    struct_seq: str,
    aligner: Optional[PairwiseAligner] = None,
) -> Tuple[List[int], float]:
    """Align PS4 sequence to structure sequence and return mapping using PairwiseAligner."""
    # Clean PS4 sequence
    ps4_seq = ps4_seq.upper()
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    ps4_clean = "".join(ch if ch in valid_aas else "X" for ch in ps4_seq)

    # If either sequence is empty, bail
    if len(ps4_clean) == 0 or len(struct_seq) == 0:
        return [-1] * len(ps4_seq), 0.0

    if aligner is None:
        aligner = _get_aligner()

    alignments = aligner.align(ps4_clean, struct_seq)
    if len(alignments) == 0:
        return [-1] * len(ps4_seq), 0.0

    alignment = alignments[0]
    query_blocks, target_blocks = alignment.aligned  # coordinates of aligned runs

    # Reconstruct gapped alignment to mirror pairwise2 iteration
    aligned_pairs: List[Tuple[str, str]] = []
    q_pos = 0
    t_pos = 0
    for (qs, qe), (ts, te) in zip(query_blocks, target_blocks):
        if qs > q_pos:
            for _ in range(qs - q_pos):
                aligned_pairs.append((ps4_clean[q_pos], "-"))
                q_pos += 1
        if ts > t_pos:
            for _ in range(ts - t_pos):
                aligned_pairs.append(("-", struct_seq[t_pos]))
                t_pos += 1
        for _ in range(qe - qs):
            aligned_pairs.append((ps4_clean[q_pos], struct_seq[t_pos]))
            q_pos += 1
            t_pos += 1

    # Trailing gaps, if any
    while q_pos < len(ps4_clean):
        aligned_pairs.append((ps4_clean[q_pos], "-"))
        q_pos += 1
    while t_pos < len(struct_seq):
        aligned_pairs.append(("-", struct_seq[t_pos]))
        t_pos += 1

    # Build mapping
    L_ps4 = len(ps4_seq)
    map_seq_to_struct = [-1] * L_ps4

    i_ps4 = 0
    j_struct = 0

    for a_ch, b_ch in aligned_pairs:
        if a_ch != "-" and b_ch != "-":
            # position i_ps4 in PS4 maps to j_struct in struct
            if i_ps4 < L_ps4:
                map_seq_to_struct[i_ps4] = j_struct
            i_ps4 += 1
            j_struct += 1
        elif a_ch != "-" and b_ch == "-":
            # gap in structure: PS4 residue has no structure here
            i_ps4 += 1
        elif a_ch == "-" and b_ch != "-":
            # gap in PS4: extra structure residue (ignored)
            j_struct += 1
        else:
            continue

    mapped = sum(1 for idx in map_seq_to_struct if idx != -1)
    coverage = mapped / max(1, L_ps4)

    return map_seq_to_struct, coverage


def get_polymer_label_ids(mmcif_dict: MMCIF2Dict) -> set:
    """Return the set of label_asym_ids that correspond to polymer entities."""
    strand_field = mmcif_dict.get("_entity_poly.pdbx_strand_id", [])
    if isinstance(strand_field, str):
        strand_field = [strand_field]

    polymer_labels = set()
    for entry in strand_field:
        # entries can be comma-separated lists of chain IDs
        for label in str(entry).replace(" ", "").split(","):
            if label:
                polymer_labels.add(label)
    return polymer_labels


def build_auth_to_label_map(mmcif_dict: MMCIF2Dict) -> dict:
    """Map author chain IDs to one representative label chain ID."""
    labels = mmcif_dict.get("_atom_site.label_asym_id", [])
    auths = mmcif_dict.get("_atom_site.auth_asym_id", [])
    auth_to_label = {}
    for label, auth in zip(labels, auths):
        if auth not in auth_to_label:
            auth_to_label[auth] = label
    return auth_to_label


def build_label_to_auth_map(mmcif_dict: MMCIF2Dict) -> dict:
    """Map label chain IDs to their author chain ID."""
    labels = mmcif_dict.get("_atom_site.label_asym_id", [])
    auths = mmcif_dict.get("_atom_site.auth_asym_id", [])
    label_to_auth = {}
    for label, auth in zip(labels, auths):
        if label not in label_to_auth:
            label_to_auth[label] = auth
    return label_to_auth


def select_chain_for_sequence(
    structure,
    requested_chain_id: str,
    ps4_seq: str,
    polymer_labels: set,
    min_coverage: float,
    auth_to_label: dict,
    label_to_auth: dict,
) -> Tuple[str, List, List[int]]:
    """Find the best chain to align ps4_seq against, with fallbacks.

    Preference order:
      1) requested_chain_id (as label ID)
      2) mapped label ID if requested is an author ID
      3) best-matching polymer chain by alignment coverage
    """
    tried = set()
    attempt_order: List[str] = [requested_chain_id]

    mapped_label = auth_to_label.get(requested_chain_id)
    if mapped_label and mapped_label not in attempt_order:
        attempt_order.append(mapped_label)

    auth_from_label = label_to_auth.get(requested_chain_id)
    if auth_from_label:
        mapped_from_auth = auth_to_label.get(auth_from_label)
        if mapped_from_auth and mapped_from_auth not in attempt_order:
            attempt_order.append(mapped_from_auth)

    best_candidate = None  # (chain_id, residues, mapping)
    best_cov = -1.0
    failures = []

    def evaluate_chain(chain_id: str):
        residues = get_chain_residues_with_ca(structure, chain_id)
        struct_seq = residues_to_sequence(residues)
        mapping, cov = align_ps4_to_structure(ps4_seq, struct_seq)
        return residues, mapping, cov

    for cid in attempt_order:
        if cid in tried:
            continue
        tried.add(cid)
        try:
            residues, mapping, cov = evaluate_chain(cid)
        except Exception as e:
            failures.append(f"{cid}: {e}")
            continue
        if cov >= min_coverage and any(m != -1 for m in mapping):
            return cid, residues, mapping
        failures.append(f"{cid}: coverage {cov:.3f}")
        if cov > best_cov:
            best_candidate = (cid, residues, mapping)
            best_cov = cov

    # fallback: scan polymer chains for best coverage
    for cid in sorted(polymer_labels):
        if cid in tried:
            continue
        tried.add(cid)
        try:
            residues, mapping, cov = evaluate_chain(cid)
        except Exception as e:
            failures.append(f"{cid}: {e}")
            continue
        if cov > best_cov:
            best_candidate = (cid, residues, mapping)
            best_cov = cov

    if best_candidate and best_cov >= min_coverage:
        return best_candidate

    failure_msg = "; ".join(failures) if failures else "no viable chains"
    raise ValueError(
        f"No chain meets coverage threshold ({min_coverage}) for requested '{requested_chain_id}': "
        f"{failure_msg}"
    )


def get_ca_coordinates_with_mask_aligned(
    structure,
    chain_id: str,
    ps4_seq: str,
    min_coverage: float = 0.5,
    polymer_labels: set = None,
    auth_to_label: dict = None,
    label_to_auth: dict = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Extract Cα coords & valid-mask for a PS4 sequence using alignment.

    Steps:
      1) Choose the best chain (requested → mapped author → best polymer) via alignment.
      2) Convert residues to struct_seq.
      3) Align ps4_seq to struct_seq; build PS4→structure index mapping.
      4) For PS4 positions that map to a structure index, take CA coords.

    Returns:
      ca_coords:   [L', 3] float32, Cα coords for positions with structure
      valid_mask:  [L] bool, True where PS4 pos has structure coords
      valid_indices: list of PS4 indices (0-based) with structure coords
    """
    polymer_labels = polymer_labels or set()
    auth_to_label = auth_to_label or {}
    label_to_auth = label_to_auth or {}

    selected_chain_id, residues, map_seq_to_struct = select_chain_for_sequence(
        structure,
        chain_id,
        ps4_seq,
        polymer_labels,
        min_coverage,
        auth_to_label,
        label_to_auth,
    )

    if selected_chain_id != chain_id:
        pass
        # print(
        #     f"[INFO] Using chain '{selected_chain_id}' for {structure.id} "
        #     f"(requested '{chain_id}') based on alignment coverage"
        # )

    L = len(ps4_seq)
    coverage = sum(1 for idx in map_seq_to_struct if idx != -1) / max(1, L)
    valid_mask = np.zeros(L, dtype=bool)
    valid_indices: List[int] = []
    ca_coords: List[np.ndarray] = []

    for i_ps4, j_struct in enumerate(map_seq_to_struct):
        if j_struct < 0:
            continue
        if j_struct >= len(residues):
            continue  # safety
        res = residues[j_struct]
        if "CA" not in res:
            continue
        valid_mask[i_ps4] = True
        valid_indices.append(i_ps4)
        ca_coords.append(res["CA"].coord)

    if len(ca_coords) == 0 or coverage < min_coverage:
        raise ValueError(
            f"Insufficient mapping coverage for chain '{chain_id}' in {structure.id}: "
            f"coverage={coverage:.3f}, mapped_positions={len(ca_coords)}/{L}"
        )

    return np.asarray(ca_coords, dtype=np.float32), valid_mask, valid_indices


def compute_contact_map(
    ca_coords: np.ndarray,
    valid_indices: List[int],
    cutoff: float = 8.0,
    min_seq_sep: int = 4,
) -> torch.Tensor:
    """Compute binary contact map from Cα coordinates.

    The min_seq_sep is applied based on original sequence positions, not
    the positions in the reduced contact map.

    Args:
        ca_coords: [L', 3] np.ndarray of Cα coords
        valid_indices: list of PS4 indices corresponding to each coord row
        cutoff: distance cutoff in Angstroms
        min_seq_sep: minimum |i - j| in PS4 index space to consider a contact

    Returns:
        contact_map: [L', L'] torch.FloatTensor with 0.0 / 1.0
    """
    L_prime = ca_coords.shape[0]

    # Pairwise squared distances
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # [L', L', 3]
    dist2 = np.sum(diff * diff, axis=-1)                  # [L', L']
    cutoff2 = cutoff * cutoff

    contact_map = dist2 <= cutoff2  # bool [L', L']

    # Apply |i-j| >= min_seq_sep in original PS4 index space
    valid_indices_arr = np.asarray(valid_indices, dtype=np.int32)
    seq_i = valid_indices_arr[:, None]   # [L', 1]
    seq_j = valid_indices_arr[None, :]   # [1, L']
    seq_sep = np.abs(seq_i - seq_j)      # [L', L']

    mask_far = seq_sep >= min_seq_sep
    contact_map &= mask_far

    return torch.from_numpy(contact_map.astype(np.float32))


def process_entry(task):
    """Worker for contact map computation."""
    (
        chain_id,
        ps4_seq,
        first_res,
        cif_path_str,
        output_path_str,
        cutoff,
        min_seq_sep,
        min_coverage,
    ) = task
    output_path = Path(output_path_str)
    seq_len = len(ps4_seq)
    pdb_id = chain_id[:4].lower()
    chain_letter = chain_id[4]

    try:
        mmcif_dict = _get_mmcif_dict(cif_path_str)
        polymer_labels = get_polymer_label_ids(mmcif_dict)
        auth_to_label = build_auth_to_label_map(mmcif_dict)
        label_to_auth = build_label_to_auth_map(mmcif_dict)

        structure = _get_structure(pdb_id, cif_path_str)

        ca_coords, valid_mask, valid_indices = get_ca_coordinates_with_mask_aligned(
            structure,
            chain_letter,
            ps4_seq,
            min_coverage=min_coverage,
            polymer_labels=polymer_labels,
            auth_to_label=auth_to_label,
            label_to_auth=label_to_auth,
        )

        contact_map = compute_contact_map(
            ca_coords,
            valid_indices,
            cutoff=cutoff,
            min_seq_sep=min_seq_sep,
        )

        output_data = {
            "contact_map": contact_map,
            "valid_mask": torch.from_numpy(valid_mask),
            "valid_indices": torch.tensor(valid_indices, dtype=torch.long),
            "seq_len": seq_len,
        }
        torch.save(output_data, output_path)

        return {
            "status": "ok",
            "chain_id": chain_id,
            "seq_len": seq_len,
            "mapped": len(valid_indices),
        }
    except Exception as e:
        return {
            "status": "error",
            "chain_id": chain_id,
            "seq_len": seq_len,
            "msg": f"{e} (first_res={first_res}, seq_len={seq_len})",
        }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} entries from {args.csv_path}")

    total_residues = 0
    missing_residues = 0
    skipped_entries = 0
    existing_outputs = 0
    saved_entries = 0
    overwritten_outputs = 0
    tasks = []
    for _, row in df.iterrows():
        chain_id = str(row["chain_id"])
        ps4_seq = str(row["input"])
        first_res = int(row.get("first_res", 1))

        output_path = output_dir / f"{chain_id}.pt"
        if output_path.exists():
            if not args.overwrite:
                existing_outputs += 1
                continue
            overwritten_outputs += 1

        pdb_id = chain_id[:4].lower()
        cif_path = Path(args.mmcif_dir) / f"{pdb_id}.cif"
        if not cif_path.exists():
            print(
                f"[WARN] CIF file not found: {cif_path} "
                f"(for chain_id: {chain_id}), skipping"
            )
            skipped_entries += 1
            continue

        tasks.append(
            (
                chain_id,
                ps4_seq,
                first_res,
                str(cif_path),
                str(output_path),
                args.cutoff,
                args.min_seq_sep,
                args.min_coverage,
            )
        )

    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 1)

    if num_workers == 1:
        iterator = (process_entry(t) for t in tasks)
        for res in tqdm(iterator, total=len(tasks), desc="Computing contact maps"):
            if res["status"] == "ok":
                saved_entries += 1
                total_residues += res["seq_len"]
                missing_residues += (res["seq_len"] - res["mapped"])
            else:
                skipped_entries += 1
                print(f"[WARN] Failed to extract CA coords for {res['chain_id']}: {res['msg']}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_entry, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Computing contact maps"):
                res = fut.result()
                if res["status"] == "ok":
                    saved_entries += 1
                    total_residues += res["seq_len"]
                    missing_residues += (res["seq_len"] - res["mapped"])
                else:
                    skipped_entries += 1
                    print(f"[WARN] Failed to extract CA coords for {res['chain_id']}: {res['msg']}")

    print(f"\nSaved {saved_entries} contact maps to {output_dir}")
    if overwritten_outputs:
        print(f"Overwrote existing outputs: {overwritten_outputs}")
    print(f"Skipped existing outputs: {existing_outputs}")
    print(f"Skipped entries: {skipped_entries}")
    print(f"Total residues processed: {total_residues}")
    if total_residues > 0:
        print(
            f"Missing residues among processed entries: {missing_residues} "
            f"({100 * missing_residues / max(1, total_residues):.2f}%)"
        )


if __name__ == "__main__":
    main()
