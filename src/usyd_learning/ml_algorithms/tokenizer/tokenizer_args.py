from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
import re

from usyd_learning.ml_utils import KeyValueArgs, dict_exists, dict_get

@dataclass
class TokenizerArgs(KeyValueArgs):
    """
    Tokenizer arguments (minimal, focused on basic_english tokenizer).
    """

    # basic identity
    tokenizer_type: str = "basic_english"  # only basic_english supported for now

    # simple preprocessing options
    lowercase: bool = True
    remove_punct: bool = True

    # special tokens / vocab
    keep_special_tokens: List[str] | None = None
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    # length control
    max_length: int = 512
    truncation: bool = True
    padding: bool | str = False

    # vocab training / mapping (optional)
    vocab: Any = None  # user can place a dict token->id here
    vocab_size: int = 30000
    min_frequency: int = 1

    # runtime
    tokenizer_obj: Optional[Callable[[str], List[str]]] = None
    verbose: bool = False

    def __init__(self, config_dict: dict | None = None, is_clone_dict: bool = False):
        super().__init__(config_dict, is_clone_dict)

        # allow config under root or under a `tokenizer` subsection
        if config_dict is not None and dict_exists(config_dict, "tokenizer"):
            self.set_args(dict_get(config_dict, "tokenizer"), is_clone_dict)

        self.tokenizer_type = self.get("type", self.tokenizer_type)
        self.lowercase = self.get("lowercase", self.lowercase)
        self.remove_punct = self.get("remove_punct", self.remove_punct)
        self.keep_special_tokens = self.get("keep_special_tokens", self.keep_special_tokens)

        self.max_length = self.get("max_length", self.max_length)
        self.truncation = self.get("truncation", self.truncation)
        self.padding = self.get("padding", self.padding)

        self.pad_token = self.get("pad_token", self.pad_token)
        self.unk_token = self.get("unk_token", self.unk_token)

        self.vocab = self.get("vocab", self.vocab)
        self.vocab_size = self.get("vocab_size", self.vocab_size)
        self.min_frequency = self.get("min_frequency", self.min_frequency)

        self.tokenizer_obj = self.get("tokenizer_obj", self.tokenizer_obj)
        self.verbose = self.get("verbose", self.verbose)

    def build_tokenizer(self) -> Callable[[str], List[str]]:
        """
        Build or return a tokenizer callable. For now supports `basic_english` only.
        The returned callable accepts a string and returns a list of token strings.
        """
        if self.tokenizer_obj is not None:
            return self.tokenizer_obj

        if self.tokenizer_type != "basic_english":
            raise ValueError(f"Tokenizer type '{self.tokenizer_type}' not supported; only 'basic_english' is available")

        # basic_english behavior: lowercasing (optional), extract word tokens (alphanumeric + underscore)
        word_re = re.compile(r"\w+", flags=re.UNICODE)

        def _tokenize(text: str) -> List[str]:
            if text is None:
                return []
            if not isinstance(text, str):
                text = str(text)

            if self.lowercase:
                text_proc = text.lower()
            else:
                text_proc = text

            if self.remove_punct:
                tokens = word_re.findall(text_proc)
            else:
                # simple whitespace split as fallback
                tokens = text_proc.split()

            if self.truncation and self.max_length is not None:
                tokens = tokens[: self.max_length]

            return tokens

        self.tokenizer_obj = _tokenize
        return _tokenize

    def encode(self, text: str) -> List[int]:
        """
        A tiny helper that tokenizes then maps tokens to ids using `vocab` if provided.
        If `vocab` is None, returns token strings' hashes as placeholder ids.
        """
        tok = self.build_tokenizer()
        tokens = tok(text)
        if self.vocab is None:
            # fallback: produce stable hash-based ids (not for production)
            return [abs(hash(t)) % (10 ** 9) for t in tokens]

        out = []
        for t in tokens:
            out.append(self.vocab.get(t, self.vocab.get(self.unk_token, 0)))
        return out

class TokenizerArgs:

    def __init__(self):
        pass

    def create_args(self):
        pass