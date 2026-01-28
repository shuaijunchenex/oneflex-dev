from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

"""
Dataset loader for AG_NEWS (train/test splits)
"""


class DatasetLoader_Agnews(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        train_split = getattr(args, "train_split", "train") or getattr(args, "split", "train")
        test_split = getattr(args, "test_split", "test")

        if is_download:
            try:
                _ = AG_NEWS(root=root, split=train_split)
            except Exception:
                pass

        self._dataset = AG_NEWS(root=root, split=train_split)
        self._test_dataset = AG_NEWS(root=root, split=test_split)

        hf_tokenizer = getattr(args, "tokenizer")
        args.vocab_size = getattr(hf_tokenizer, "vocab_size", None)

        collate = partial(
            DatasetLoaderUtil.text_collate_fn_hf,
            hf_tokenizer=hf_tokenizer,
            max_len=getattr(args, "max_len", 256),
            label_map={1: 0, 2: 1, 3: 2, 4: 3},
            normalize_int_labels=False,
        )

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

        self.task_type = "nlp"
        try:
            self.data_sample_num = len(self._dataset)
        except Exception:
            self.data_sample_num = None

        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )

        return
