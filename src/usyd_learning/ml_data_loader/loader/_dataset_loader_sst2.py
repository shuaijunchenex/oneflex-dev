from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader
from torchtext.datasets import SST2

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

"""
Dataset loader for SST-2 (GLUE)
"""


class DatasetLoader_SST2(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _warmup_download(self, root: str, splits: tuple[str, ...]):
        for sp in splits:
            try:
                it = iter(SST2(root=root, split=sp))
                next(it)
            except StopIteration:
                continue
            except Exception:
                continue

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        train_split = getattr(args, "train_split", "train")
        test_split = getattr(args, "test_split", "dev")

        if is_download:
            self._warmup_download(root, (train_split, test_split))

        self._dataset = SST2(root=root, split=train_split)
        self._test_dataset = SST2(root=root, split=test_split)

        hf_tokenizer = getattr(args, "tokenizer")
        args.vocab_size = getattr(hf_tokenizer, "vocab_size", None)

        collate = partial(
            DatasetLoaderUtil.text_collate_fn_hf,
            hf_tokenizer=hf_tokenizer,
            max_len=getattr(args, "max_len", 256),
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

    def get_dataset(self):
        if self._data_loader is not None:
            return self._data_loader.dataset
        raise ValueError("ERROR: DatasetLoader's data_loader is None.")
