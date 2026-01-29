from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader
from torchtext.datasets import MRPC

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil
from ...ml_algorithms.tokenizer_builder import TokenizerBuilder

"""
Dataset loader for MRPC (GLUE)
"""


class DatasetLoader_MRPC(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _warmup_download(self, root: str, splits: tuple[str, ...]):
        for sp in splits:
            try:
                # map common aliases to supported splits
                use_sp = "test" if sp in ("valid", "validation", "dev", "valid_matched", "valid_mismatched") else sp
                it = iter(MRPC(root=root, split=use_sp))
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
        test_split = getattr(args, "test_split", "test")
        # map common aliases to torchtext MRPC-supported splits ('train', 'test')
        if test_split in ("valid", "validation", "dev"):
            test_split = "test"
        combine_fn = getattr(args, "text_pair_combine_fn", None)

        if is_download:
            self._warmup_download(root, (train_split, test_split))

        self._dataset = MRPC(root=root, split=train_split)
        self._test_dataset = MRPC(root=root, split=test_split)

        tokenizer = getattr(args, "tokenizer")
        self.vocab = getattr(args, "vocab", None)
        if self.vocab is None:
            self.vocab = TokenizerBuilder.build_vocab(self._dataset, tokenizer)
            
        args.vocab_size = len(self.vocab)

        collate = partial(
            DatasetLoaderUtil.text_pair_collate_fn,
            tokenizer=tokenizer,
            vocab=self.vocab,
            combine_fn=combine_fn,
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
