from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional

from ...ml_utils import KeyValueArgs


class Tokenizer(ABC):
    """
    Abstract tokenizer class. Mirrors DatasetLoader style: a create(args) entrypoint
    and concrete implementations implement _create_inner to build internal tokenizer.

    Expected responsibilities of concrete tokenizers:
      - implement _create_inner(args) to initialize tokenizer state
      - provide tokenize(text)->List[str] and encode(text)->List[int]
      - optionally implement build_vocab(data_iter)
    """

    def __init__(self):
        self._args: KeyValueArgs | None = None
        self._tokenize_fn: Optional[Callable[[str], List[str]]] = None
        self._encode_fn: Optional[Callable[[str], List[int]]] = None
        self._is_created: bool = False
        return

    @property
    def args(self) -> KeyValueArgs | None:
        return self._args

    @property
    def is_created(self) -> bool:
        return self._is_created

    def create(self, args: KeyValueArgs) -> "Tokenizer":
        """
        Create/initialize tokenizer from args (KeyValueArgs or subclass).
        Concrete implementation should implement _create_inner(args).
        """
        self._args = args
        self._create_inner(args)
        self._is_created = True
        return self

    @abstractmethod
    def _create_inner(self, args: KeyValueArgs) -> None:
        """Do real initialization here"""
        pass

    def tokenize(self, text: str) -> List[str]:
        if not self._is_created:
            raise RuntimeError("Tokenizer not created. Call create(args) first.")
        if self._tokenize_fn is None:
            raise NotImplementedError("tokenize not implemented by concrete tokenizer")
        return self._tokenize_fn(text)

    def encode(self, text: str) -> List[int]:
        if not self._is_created:
            raise RuntimeError("Tokenizer not created. Call create(args) first.")
        if self._encode_fn is not None:
            return self._encode_fn(text)
        # default: map tokens to hash-based ids if no encode_fn
        toks = self.tokenize(text)
        return [abs(hash(t)) % (10 ** 9) for t in toks]

    def set_tokenize_fn(self, fn: Callable[[str], List[str]]):
        self._tokenize_fn = fn

    def set_encode_fn(self, fn: Callable[[str], List[int]]):
        self._encode_fn = fn

    def build_vocab(self, data_iter: Iterable, **kwargs) -> Any:
        """
        Optional: build vocab from an iterable of samples. Concrete implementations
        may override.
        """
        raise NotImplementedError()
