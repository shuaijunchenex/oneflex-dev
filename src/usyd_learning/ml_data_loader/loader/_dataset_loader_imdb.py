from __future__ import annotations


from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil
from torch.utils.data import DataLoader #
from torchtext.datasets import IMDB
from torch.utils.data import IterableDataset
from functools import partial
from datasets import load_dataset

import torch
'''
Dataset loader for imdb
'''
class DatasetLoader_Imdb(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _warmup_download(self, root: str):
        for sp in ("train", "test"):
            it = iter(IMDB(root=root, split=sp))
            try:
                next(it)
            except StopIteration:
                pass

    def _load_imdb_hf(self):
        ds = load_dataset("imdb")
        # 统一成 (label_str, text) 形式，保持与 torchtext.IMDB 对齐
        train = [("pos" if int(x["label"]) == 1 else "neg", x["text"]) for x in ds["train"]]
        test = [("pos" if int(x["label"]) == 1 else "neg", x["text"]) for x in ds["test"]]
        return train, test

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        
        self._dataset, self._test_dataset = self._load_imdb_hf()

        # self._dataset = list(IMDB(root=root, split="train"))
        # self._test_dataset = list(IMDB(root=root, split="test"))

        hf_tokenizer = getattr(args, "tokenizer")
        args.vocab_size = getattr(hf_tokenizer, "vocab_size", None)

        if is_download:
            self._warmup_download(root)

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle= True,
            num_workers=num_workers,
            collate_fn=partial(
                DatasetLoaderUtil.text_collate_fn_hf,
                hf_tokenizer=hf_tokenizer,
                max_len=getattr(args, "max_len", 256),
                label_map={"neg": 0, "pos": 1},
            ),
        )

        self.data_sample_num = 25000
        self.task_type = "nlp"
        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(
                DatasetLoaderUtil.text_collate_fn_hf,
                hf_tokenizer=hf_tokenizer,
                max_len=getattr(args, "max_len", 256),
                label_map={"neg": 0, "pos": 1},
            ),
        )
        return

    def get_dataset(self) -> DataLoader:
        if self._data_loader is not None:
            return self._data_loader.dataset
        else:
            raise ValueError("ERROR: DatasetLoader's data_loader is None.")
