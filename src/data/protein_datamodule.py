from typing import Any, Dict, Optional, Tuple, List
import csv
import random

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split

from src.data.constants import CANONICAL_AA, INPUT_ALPHABET

class ProteinDataset(Dataset):
    def __init__(self, data: List[Dict], input_vocab: Dict[str, int], target_vocab: Dict[str, int]):
        self.data = data
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        input_seq = [self.input_vocab[c] for c in row['input']]
        target_seq = [self.target_vocab[c] for c in row['dssp8']]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.lengths = self._compute_lengths(dataset)

    def _get_length_attr(self, dataset):
        if hasattr(dataset, "sequence_lengths"):
            return dataset.sequence_lengths
        if hasattr(dataset, "lengths"):
            return dataset.lengths
        return None

    def _compute_lengths(self, dataset):
        lengths = []
        # Optimization to avoid full dataset iteration with __getitem__
        if isinstance(dataset, Subset):
            parent = dataset.dataset
            attr_lengths = self._get_length_attr(parent)
            if attr_lengths is not None:
                return [attr_lengths[idx] for idx in dataset.indices]
            if hasattr(parent, "data"):
                return [len(parent.data[idx]["input"]) for idx in dataset.indices]
            # Fall through to generic path if no shortcuts available
        else:
            attr_lengths = self._get_length_attr(dataset)
            if attr_lengths is not None:
                return list(attr_lengths)
            if hasattr(dataset, "data"):
                return [len(x["input"]) for x in dataset.data]

        for i in range(len(dataset)):
            lengths.append(len(dataset[i][0]))
        return lengths

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        # Sort indices by sequence length
        indices.sort(key=lambda i: self.lengths[i])
        
        # Create batches
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches.pop()
            
        if self.shuffle:
            # Shuffle the batches (keeps items of similar length together in a batch)
            random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class ProteinDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/ps4_data.csv",
        batch_size: int = 32,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.input_vocab: Dict[str, int] = {}
        self.target_vocab: Dict[str, int] = {}

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train and self.data_val and self.data_test:
            return

        # Read Data as a list of dicts
        with open(self.hparams.data_dir, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)

        # Get all unique characters for Amino Acids and DSSP8
        target_chars = set()
        for row in data:
            # Normalize input sequence: uppercase + map non-canonical to 'X'
            raw_seq = row["input"].upper()
            normalized_seq = [c if c in CANONICAL_AA else "X" for c in raw_seq]
            row["input"] = "".join(normalized_seq)
            target_chars.update(row["dssp8"].upper())

        # Map the letters to integers: 0 for padding, 1..N for actual chars
        self.input_vocab = {c: i + 1 for i, c in enumerate(INPUT_ALPHABET)}
        self.input_vocab["<PAD>"] = 0

        self.target_vocab = {c: i + 1 for i, c in enumerate(sorted(list(target_chars)))}
        self.target_vocab["<PAD>"] = 0


        # Compute Train, Val, Test Sizes
        total_len = len(data)
        train_len = int(total_len * self.hparams.train_val_test_split[0])
        val_len = int(total_len * self.hparams.train_val_test_split[1])
        test_len = total_len - train_len - val_len


        # Split Train, Val, Test
        full_dataset = ProteinDataset(data, self.input_vocab, self.target_vocab)
        self.data_train, self.data_val, self.data_test = random_split(
        full_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
        )

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
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
        
    @staticmethod
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        
        # Pad sequences
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
        
        return inputs_padded, targets_padded
