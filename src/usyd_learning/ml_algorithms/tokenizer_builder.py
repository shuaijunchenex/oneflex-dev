from __future__ import annotations
import re
import torch
import torch.optim as optim
from typing import Callable, Any, Optional, Dict, List
from ..ml_utils import Handlers
from torch.nn.utils.rnn import pad_sequence
#from transformers import AutoTokenizer
from torchtext.vocab import build_vocab_from_iterator

class TokenizerBuilder(Handlers):
    """
    config 示例：
    {
      "tokenizer": {
        "type": "basic_english",   # 或 "bert"
        "lower": true,             # 仅对 basic_english
        "pretrained": "bert-base-uncased",  # 仅对 bert
        "use_fast": true,          # 仅对 bert
        "return_type": "tokens",   # tokens | ids  (bert 默认 tokens)
        "max_len": 256             # 仅对 bert 且 return_type=ids 时用于截断
      }
    }
    build() 返回一个可调用对象：
      - basic_english: fn(text) -> List[str]
      - bert:
          return_type='tokens' -> tok.tokenize(text) -> List[str]
          return_type='ids'    -> fn(text) -> List[int]  (包含 special tokens，按 max_len 截断)
    """

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__()
        self.config = config_dict.get("tokenizer", config_dict)
        self._tokenizer_fn: Optional[Callable[[str], Any]] = None
        self.meta: Dict[str, Any] = {}  # 可选：记录 pad_id / hf_tokenizer 等

        self.register_handler("basic_english", self.__build_basic_english)
        self.register_handler("bert",           self.__build_bert)

    def build(self) -> Callable[[str], Any]:
        t = str(self.config.get("type", "basic_english")).lower()
        fn = self.invoke_handler(t)
        if fn is None:
            raise ValueError(f"Unsupported tokenizer type: {t}")
        self._tokenizer_fn = fn
        return fn

    @staticmethod
    def build_vocab(data_set, tokenizer, min_freq=2, max_vocab=30000):
        from collections import Counter
        from torchtext.vocab import Vocab
        
        train_pairs = list(data_set)

        PAD, CLS, UNK = "<pad>", "<cls>", "<unk>"
        specials = [PAD, CLS, UNK]
        counter = Counter()

        for _, text in train_pairs:
            counter.update(tokenizer(text))

        cutoff = max_vocab - len(specials)
        most_common = counter.most_common(cutoff)
        counter = Counter(dict(most_common))

        def yield_tokens():
            for _, text in train_pairs:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(
            yield_tokens(),
            min_freq=min_freq,
            specials=specials,
            max_tokens=max_vocab
        )
        
        vocab.set_default_index(vocab[UNK])

        return vocab

    # def get_imdb_iter(root: str, split: str):
    #     # Try new API first; fall back to legacy if TypeError (your error)
    #     try:
    #         it = _imdb_iter_new(root, split)
    #         # ensure it's materializable
    #         it = list(it)  # download on first call
    #         return it
    #     except TypeError:
    #         # legacy path
    #         return list(_imdb_iter_legacy(root, split))

    # def _imdb_iter_new(root: str, split: str):
    #     # New API: torchtext.datasets.IMDB(root=..., split='train'/'test')
    #     from torchtext.datasets import IMDB
    #     return IMDB(root=root, split=split)

    # -------- builders --------
    def __build_basic_english(self) -> Callable[[str], Any]:
        try:
            from torchtext.data.utils import get_tokenizer
        except Exception as e:
            raise ImportError("torchtext is required for 'basic_english' tokenizer.") from e

        lower = self.config.get("lower", True)
        base = get_tokenizer("basic_english")
        return (lambda s: base(s.lower())) if lower else base

    def __build_bert(self) -> Callable[[str], Any]:
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise ImportError("transformers is required for 'bert' tokenizer.") from e

        name = self.config.get("pretrained", "bert-base-uncased")
        use_fast = self.config.get("use_fast", True)
        return_type = str(self.config.get("return_type", "tokens")).lower()
        max_len = int(self.config.get("max_len", 512))

        tok = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
        self.meta["hf_tokenizer"] = tok
        if tok.pad_token_id is not None:
            self.meta["pad_id"] = int(tok.pad_token_id)

        if return_type == "ids":
            def fn_ids(s: str):
                out = tok(s, add_special_tokens=True, truncation=True, max_length=max_len)
                return list(map(int, out["input_ids"]))
            return fn_ids
        else:
            return tok.tokenize