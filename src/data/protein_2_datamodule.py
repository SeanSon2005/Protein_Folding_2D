import csv
import gc
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.protein_datamodule import BucketBatchSampler

@dataclass(frozen=True)
class ShardLocation:
    """
    A pointer to "where a chain lives" inside a shard file.
    args:
        shard_path: which .pt file.
        entry_idx:  which index in shard_obj["embeddings"].
        length:     number of residues (aka tokens) in that chain.
    """
    shard_path: str
    entry_idx: int
    length: int

@dataclass(frozen=True)
class SampleMetadata:
    """Metadata needed to fetch a sample lazily => ShardLocation + labels"""
    chain_id: str
    shard_path: str
    entry_idx: int
    length: int
    dssp8: str


class ShardedESM2Dataset(Dataset):
    """Dataset that streams pre-computed ESM2 embeddings from large shard files."""
    def __init__(
        self,
        samples: Sequence[SampleMetadata],
        target_vocab: Dict[str, int],
        shard_cache_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.samples = list(samples)
        self.target_vocab = target_vocab
        self.sequence_lengths: List[int] = [sample.length for sample in self.samples]
        self.dtype = dtype
        self.shard_cache_size = max(1, shard_cache_size)
        self._shard_cache: "OrderedDict[str, List[torch.Tensor]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_shard(self, shard_path: str) -> List[torch.Tensor]:
        shard_path = str(shard_path)
        if shard_path in self._shard_cache:
            self._shard_cache.move_to_end(shard_path)
            return self._shard_cache[shard_path]

        shard_obj = torch.load(shard_path, map_location="cpu", mmap=True)
        embeddings: List[torch.Tensor] = shard_obj["embeddings"]
        del shard_obj

        self._shard_cache[shard_path] = embeddings
        self._shard_cache.move_to_end(shard_path)
        if len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)

        return embeddings

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        shard_embeddings = self._load_shard(sample.shard_path)
        token_embeddings = shard_embeddings[sample.entry_idx].to(dtype=self.dtype)

        if token_embeddings.shape[0] != sample.length:
            raise ValueError(
                f"Embedding length mismatch for chain {sample.chain_id}: "
                f"{token_embeddings.shape[0]} vs expected {sample.length}"
            )

        target_tensor = torch.tensor(
            [self.target_vocab[c] for c in sample.dssp8],
            dtype=torch.long,
        )

        if token_embeddings.shape[0] != target_tensor.shape[0]:
            raise ValueError(
                f"Target length mismatch for chain {sample.chain_id}: "
                f"{target_tensor.shape[0]} vs embeddings {token_embeddings.shape[0]}"
            )

        return token_embeddings, target_tensor


class ProteinDataModule2(LightningDataModule):
    """Lightning DataModule that serves sharded ESM2 embeddings with lazy loading."""

    def __init__(
        self,
        metadata_csv: str = "data/ps4_data.csv",
        embeddings_dir: str = "/esm2_token_embeddings_sharded_fp16",
        batch_size: int = 4,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        num_workers: int = 0,
        pin_memory: bool = False,
        split_seed: int = 42,
        shard_cache_size: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.target_vocab: Dict[str, int] = {}

    def prepare_data(self) -> None:
        """No-op hook required by Lightning."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None and self.data_val is not None and self.data_test is not None:
            return

        metadata_rows = self._load_metadata(Path(self.hparams.metadata_csv))
        shard_index = self._build_shard_index(Path(self.hparams.embeddings_dir))

        target_chars = {char for row in metadata_rows for char in row["dssp8"]}
        if not target_chars:
            raise RuntimeError("No DSSP8 targets found in metadata.")

        self.target_vocab = {c: i + 1 for i, c in enumerate(sorted(target_chars))}
        self.target_vocab["<PAD>"] = 0

        samples = self._build_samples(metadata_rows, shard_index)
        if not samples:
            raise RuntimeError("No overlapping samples between metadata CSV and embedding shards.")

        rng = random.Random(self.hparams.split_seed)
        rng.shuffle(samples)

        train_samples, val_samples, test_samples = self._split_samples(samples)

        common_kwargs = {
            "target_vocab": self.target_vocab,
            "shard_cache_size": self.hparams.shard_cache_size,
        }
        self.data_train = ShardedESM2Dataset(train_samples, **common_kwargs)
        self.data_val = ShardedESM2Dataset(val_samples, **common_kwargs)
        self.data_test = ShardedESM2Dataset(test_samples, **common_kwargs)

    def _split_samples(
        self, samples: List[SampleMetadata]
    ) -> Tuple[List[SampleMetadata], List[SampleMetadata], List[SampleMetadata]]:
        total = len(samples)
        splits = self.hparams.train_val_test_split
        if len(splits) != 3:
            raise ValueError("train_val_test_split must contain three fractions.")

        total_frac = sum(splits)
        if total_frac <= 0:
            raise ValueError("Sum of split fractions must be positive.")

        normalized = [frac / total_frac for frac in splits]
        train_len = int(total * normalized[0])
        val_len = int(total * normalized[1])
        test_len = total - train_len - val_len

        # Ensure each requested split receives at least one element when possible.
        if train_len == 0 and normalized[0] > 0 and total >= 1:
            train_len = 1
            if val_len > 0:
                val_len = max(0, val_len - 1)
            else:
                test_len = max(0, test_len - 1)
        if val_len == 0 and normalized[1] > 0 and total - train_len >= 1:
            val_len = 1
            test_len = max(0, test_len - 1)

        train_samples = samples[:train_len]
        val_samples = samples[train_len : train_len + val_len]
        test_samples = samples[train_len + val_len :]

        return train_samples, val_samples, test_samples

    def _load_metadata(self, csv_path: Path) -> List[Dict[str, str]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        rows: List[Dict[str, str]] = []
        with csv_path.open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                chain_id = row.get("chain_id", "").strip()
                dssp8 = row.get("dssp8", "").strip().upper()
                if not chain_id or not dssp8:
                    continue
                rows.append({"chain_id": chain_id, "dssp8": dssp8})

        if not rows:
            raise RuntimeError(f"No usable rows found in {csv_path}.")

        return rows

    def _build_shard_index(self, shard_dir: Path) -> Dict[str, ShardLocation]:
        if not shard_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {shard_dir}")

        shard_paths = sorted(shard_dir.glob("*.pt"))
        if not shard_paths:
            raise RuntimeError(f"No shard files found in {shard_dir}.")

        index: Dict[str, ShardLocation] = {}
        for shard_path in shard_paths:
            shard_obj = torch.load(shard_path, map_location="cpu", mmap=True)
            chain_ids = shard_obj["chain_ids"]
            lengths_tensor = shard_obj["lengths"]
            lengths = (
                lengths_tensor.tolist()
                if hasattr(lengths_tensor, "tolist")
                else [int(l) for l in lengths_tensor]
            )
            for entry_idx, (chain_id, length) in enumerate(zip(chain_ids, lengths)):
                if chain_id in index:
                    continue
                index[chain_id] = ShardLocation(
                    shard_path=str(shard_path),
                    entry_idx=entry_idx,
                    length=int(length),
                )
            del shard_obj
            gc.collect()

        if not index:
            raise RuntimeError(f"Shard index construction failed for {shard_dir}.")

        return index

    def _build_samples(
        self,
        metadata_rows: List[Dict[str, str]],
        shard_index: Dict[str, ShardLocation],
    ) -> List[SampleMetadata]:
        samples: List[SampleMetadata] = []
        missing_chains = 0
        mismatched_lengths = 0

        for row in metadata_rows:
            chain_id = row["chain_id"]
            shard_loc = shard_index.get(chain_id)
            if shard_loc is None:
                missing_chains += 1
                continue
            dssp8 = row["dssp8"]
            if len(dssp8) != shard_loc.length:
                mismatched_lengths += 1
                continue
            samples.append(
                SampleMetadata(
                    chain_id=chain_id,
                    shard_path=shard_loc.shard_path,
                    entry_idx=shard_loc.entry_idx,
                    length=shard_loc.length,
                    dssp8=dssp8,
                )
            )

        if missing_chains:
            print(f"Dropped {missing_chains} chains missing from shard index.")
        if mismatched_lengths:
            print(f"Dropped {mismatched_lengths} chains due to length mismatch.")

        return samples

    def _bucketed_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        batch_sampler = BucketBatchSampler(
            dataset,
            self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._bucketed_loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._bucketed_loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._bucketed_loader(self.data_test, shuffle=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit/validate/test/predict`."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Save any state that should survive checkpoints."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore datamodule state."""
        pass

    @staticmethod
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
        return inputs_padded, targets_padded
    
if __name__ == "__main__":
    # Simple test to verify DataModule functionality
    data_module = ProteinDataModule2(
        metadata_csv="data/ps4_data.csv",
        embeddings_dir="data/esm2_token_embeddings_sharded_fp16_sorted",
        batch_size=8,
        num_workers=8,
        pin_memory=True,    
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")