from __future__ import annotations
from typing import Any, Iterable, List, Dict, Optional
from collections import Counter
import re

from ..tokenizer_abc import Tokenizer
from ..tokenizer_args import TokenizerArgs


class BasicEnglishTokenizer(Tokenizer):
    """A small, dependency-free basic English tokenizer implementation.

    It tokenizes by finding word character sequences (\w+), supports simple
    lowercasing and truncation, and can build a simple vocab from data.
    """

    def __init__(self):
        super().__init__()

    def _create_inner(self, args: Any) -> None:
        # Accept a TokenizerArgs or generic KeyValueArgs-like (with get())
        if isinstance(args, TokenizerArgs):
            ta = args
        else:
            # best-effort mapping from KeyValueArgs-like
            ta = TokenizerArgs()
            try:
                ta.lowercase = args.get("lowercase", ta.lowercase)
                ta.remove_punct = args.get("remove_punct", ta.remove_punct)
                ta.max_length = args.get("max_length", ta.max_length)
                ta.truncation = args.get("truncation", ta.truncation)
                ta.vocab = args.get("vocab", ta.vocab)
                ta.unk_token = args.get("unk_token", ta.unk_token)
                ta.pad_token = args.get("pad_token", ta.pad_token)
            except Exception:
                # ignore and use defaults
                pass

        word_re = re.compile(r"\w+", flags=re.UNICODE)

        def _tokenize(text: str) -> List[str]:
            if text is None:
                return []
            if not isinstance(text, str):
                text = str(text)

            proc = text.lower() if ta.lowercase else text
            if ta.remove_punct:
                toks = word_re.findall(proc)
            else:
                toks = proc.split()

            if ta.truncation and ta.max_length is not None:
                toks = toks[: ta.max_length]
            return toks

        self.set_tokenize_fn(_tokenize)

        # encode: if vocab provided, map to ids; otherwise use base class hash fallback
        if ta.vocab is not None:
            def _encode(text: str) -> List[int]:
                toks = _tokenize(text)
                return [ta.vocab.get(t, ta.vocab.get(ta.unk_token, 0)) for t in toks]

            self.set_encode_fn(_encode)

        # attach args for reference
        self._args = args

    def build_vocab(self, data_iter: Iterable, special_tokens: Optional[List[str]] = None,
                    max_size: Optional[int] = None, min_freq: int = 1) -> Dict[str, int]:
        counter = Counter()
        tok_fn = self._tokenize_fn
        if tok_fn is None:
            raise RuntimeError("tokenize function not initialized; call create(args) first")

        for sample in data_iter:
            try:
                tokens = tok_fn(sample)
            except Exception:
                tokens = tok_fn(str(sample))
            counter.update(tokens)

        # apply min frequency
        items = [(t, c) for t, c in counter.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))

        vocab: Dict[str, int] = {}
        specials = special_tokens or []
        for i, s in enumerate(specials):
            vocab[s] = i

        offset = len(specials)
        for i, (t, _) in enumerate(items):
            if max_size is not None and i + offset >= max_size:
                break
            vocab[t] = i + offset

        return vocab
