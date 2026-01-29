import torch
from typing import Any, Dict, List, Optional, Tuple

class DatasetLoaderUtil:
    """
    " DataLoader Util class
    """
    
    # torchtext datasets

    @staticmethod
    def text_collate_fn(
        batch,
        tokenizer=None,
        vocab=None,
        max_len: int = 256,
        pad_id: int = 0,
        unk_id: Optional[int] = None,
        # --- label handling ---
        label_map: Optional[Dict[Any, int]] = None,
        normalize_int_labels: bool = False,
        # --- tuple format handling ---
        tuple_format: str = "auto",  # "auto" | "label_text" | "idx_label_text"
        require_labels: bool = True,
    ):
        """
        Collate function for text datasets.

        Supports samples shaped as:
          - (label, text)                     -> tuple_format="label_text"
          - (idx, label, text)                -> tuple_format="idx_label_text"
          - dict with keys like {'label': ..., 'text': ...}
          - (idx, text) is NOT a supervised sample; if present and require_labels=True -> error

        Label mapping policy:
          - If labels are strings: you MUST provide label_map (dataset-level, stable).
          - If labels are ints:
              * default: keep as-is (no remap)
              * if normalize_int_labels=True: remap to [0..K-1] using label_map if given,
                else build a batch-local map (not recommended for training consistency).
        """
        if tokenizer is None or vocab is None:
            raise ValueError("text_collate_fn requires tokenizer and vocab.")
        if not batch:
            raise ValueError("Empty batch.")

        # choose unk_id
        if unk_id is None:
            # try common patterns
            try:
                unk_id = vocab["<unk>"]
            except Exception:
                try:
                    unk_id = vocab.get("<unk>")  # type: ignore[attr-defined]
                except Exception:
                    unk_id = pad_id  # fallback

        def _vocab_lookup(token: str) -> int:
            # robust lookup with unk fallback
            try:
                return vocab[token]
            except Exception:
                try:
                    return vocab.get(token, unk_id)  # type: ignore[attr-defined]
                except Exception:
                    return unk_id

        def _extract_label_text(sample) -> Tuple[Any, str]:
            # Dict sample
            if isinstance(sample, dict):
                label = sample.get("label", sample.get("labels", None))
                text = sample.get("text", sample.get("sentence", sample.get("content", "")))
                return label, text if text is not None else ""

            # Tuple/list sample
            if isinstance(sample, (list, tuple)):
                n = len(sample)
                if n == 0:
                    return None, ""

                fmt = tuple_format
                if fmt == "auto":
                    fmt = "idx_label_text" if n == 3 else "label_text"

                if fmt == "idx_label_text":
                    if n < 3:
                        # e.g., (idx, text) -> not supervised
                        return None, str(sample[-1]) if n >= 1 else ""
                    label = sample[1]
                    text = sample[-1]
                    return label, "" if text is None else str(text)

                # fmt == "label_text"
                if n < 2:
                    # text-only or malformed
                    return None, str(sample[0]) if n == 1 else ""
                label = sample[0]
                text = sample[-1]
                return label, "" if text is None else str(text)

            # Otherwise treat as text-only
            return None, "" if sample is None else str(sample)

        labels_texts = [_extract_label_text(s) for s in batch]
        labels, texts = zip(*labels_texts)

        if require_labels and any(l is None for l in labels):
            raise ValueError(
                "Some samples have no label (e.g., (idx,text) or missing 'label'). "
                "Set require_labels=False if you intentionally want unlabeled batches."
            )

        # tokenize + map to ids
        tokenized = [tokenizer(t) for t in texts]
        ids = [[_vocab_lookup(tok) for tok in toks] for toks in tokenized]

        # pad / truncate
        batch_max_len = min(max((len(seq) for seq in ids), default=0), max_len)
        padded: List[List[int]] = []
        for seq in ids:
            if len(seq) > batch_max_len:
                seq = seq[:batch_max_len]
            else:
                seq = seq + [pad_id] * (batch_max_len - len(seq))
            padded.append(seq)

        input_ids = torch.tensor(padded, dtype=torch.long)

        # labels tensor
        if not require_labels and all(l is None for l in labels):
            # return dummy labels (or you can return None, depending on your pipeline)
            return input_ids, None

        # Map/encode labels
        first_non_none = next((l for l in labels if l is not None), None)

        if isinstance(first_non_none, str):
            if label_map is None:
                raise ValueError(
                    "String labels detected but label_map is None. "
                    "Provide a stable dataset-level label_map, e.g., {'neg':0,'pos':1}."
                )
            mapped = [label_map[l] for l in labels]
        else:
            # integer-ish or other hashables
            if normalize_int_labels:
                if label_map is None:
                    # batch-local mapping (use with caution)
                    uniq = sorted(set(labels))
                    label_map = {l: i for i, l in enumerate(uniq)}
                mapped = [label_map[l] for l in labels]
            else:
                mapped = list(labels)

        labels_tensor = torch.tensor(mapped, dtype=torch.long)
        return input_ids, labels_tensor

    @staticmethod
    def text_collate_fn_hf(
        batch,
        hf_tokenizer=None,
        max_len: int = 256,
        # --- label handling ---
        label_map: Optional[Dict[Any, int]] = None,
        normalize_int_labels: bool = False,
        # --- tuple format handling ---
        tuple_format: str = "auto",  # "auto" | "label_text" | "text_label" | "idx_label_text"
        require_labels: bool = True,
    ):
        """
        Collate function for HuggingFace tokenizers. Returns (encodings, labels_tensor|None).
        """
        if hf_tokenizer is None:
            raise ValueError("text_collate_fn_hf requires hf_tokenizer.")
        if not batch:
            raise ValueError("Empty batch.")

        def _extract_label_text(sample) -> Tuple[Any, str]:
            if isinstance(sample, dict):
                label = sample.get("label", sample.get("labels", None))
                text = sample.get("text", sample.get("sentence", sample.get("content", "")))
                return label, "" if text is None else str(text)

            if isinstance(sample, (list, tuple)):
                n = len(sample)
                if n == 0:
                    return None, ""

                fmt = tuple_format
                if fmt == "auto":
                    fmt = "idx_label_text" if n == 3 else "label_text"

                if fmt == "idx_label_text":
                    if n < 3:
                        return None, str(sample[-1]) if n >= 1 else ""
                    label = sample[1]
                    text = sample[-1]
                    return label, "" if text is None else str(text)

                if fmt == "text_label":
                    # SST-2 format: (text, label)
                    if n < 2:
                        return None, str(sample[0]) if n == 1 else ""
                    text = sample[0]
                    label = sample[-1]
                    return label, "" if text is None else str(text)

                # "label_text" (default)
                if n < 2:
                    return None, str(sample[0]) if n == 1 else ""
                label = sample[0]
                text = sample[-1]
                return label, "" if text is None else str(text)

            return None, "" if sample is None else str(sample)

        labels_texts = [_extract_label_text(s) for s in batch]
        labels, texts = zip(*labels_texts)

        if require_labels and any(l is None for l in labels):
            raise ValueError(
                "Some samples have no label (e.g., (idx,text) or missing 'label'). "
                "Set require_labels=False if you intentionally want unlabeled batches."
            )

        enc = hf_tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if not require_labels and all(l is None for l in labels):
            return enc, None

        first_non_none = next((l for l in labels if l is not None), None)

        if isinstance(first_non_none, str):
            if label_map is None:
                raise ValueError("String labels detected but label_map is None.")
            mapped = [label_map[l] for l in labels]
        else:
            if normalize_int_labels:
                if label_map is None:
                    uniq = sorted(set(labels))
                    label_map = {l: i for i, l in enumerate(uniq)}
                mapped = [label_map[l] for l in labels]
            else:
                mapped = [int(l) for l in labels]

        labels_tensor = torch.tensor(mapped, dtype=torch.long)
        return enc, labels_tensor

    
    # @staticmethod
    # def text_collate_fn(batch, tokenizer=None, vocab=None, max_len=256, pad_id=0):
    #     """
    #     Collate function for text datasets.
    #             Converts list of samples into (padded_input_ids, labels).
    #             Supports samples shaped as:
    #                 - (label, text)
    #                 - (idx, label, text)
    #                 - dict with keys like {'label': ..., 'text': ...}
    #             Text is taken as the last element for tuple/list, or the 'text' field for dict.
    #             Label is taken as the first element, or the 'label' field for dict.

    #     Args:
    #         batch: list of (label, text) pairs
    #         tokenizer: callable, text -> list of tokens
    #         vocab: vocab object, supports vocab[token] -> id
    #         max_len: maximum sequence length (truncate if longer)
    #         pad_id: index used for padding

    #     Returns:
    #         input_ids: LongTensor [B, L]
    #         labels:    LongTensor [B]
    #     """
    #     if tokenizer is None or vocab is None:
    #         raise ValueError("text_collate_fn requires tokenizer and vocab.")

    #     def _extract_label_text(sample):
    #         # Dict sample
    #         if isinstance(sample, dict):
    #             label = sample.get("label", sample.get("labels"))
    #             text = sample.get("text", sample.get("sentence", sample.get("content")))
    #             return label, text
    #         # Tuple/list sample
    #         if isinstance(sample, (list, tuple)):
    #             if len(sample) == 0:
    #                 return None, ""
    #             label = sample[0]
    #             text = sample[-1]
    #             return label, text
    #         # Otherwise treat as text-only
    #         return None, sample

    #     labels_texts = [ _extract_label_text(s) for s in batch ]
    #     labels, texts = zip(*labels_texts)

    #     # tokenize + map to ids
    #     tokenized = [tokenizer(t) for t in texts]
    #     ids = [[vocab[token] for token in toks] for toks in tokenized]

    #     # pad / truncate
    #     batch_max_len = min(max(len(seq) for seq in ids), max_len)
    #     padded = []
    #     for seq in ids:
    #         if len(seq) > batch_max_len:
    #             seq = seq[:batch_max_len]
    #         else:
    #             seq = seq + [pad_id] * (batch_max_len - len(seq))
    #         padded.append(seq)

    #     input_ids = torch.tensor(padded, dtype=torch.long)
        
    #     # Normalize labels: strings -> ids, integers -> zero-based contiguous ids
    #     unique_labels = sorted(set(labels)) if labels else []
    #     if labels:
    #         if isinstance(labels[0], str):
    #             label_map = {l: i for i, l in enumerate(unique_labels)}
    #             labels = [label_map[l] for l in labels]
    #         else:
    #             # integer-like labels; remap to [0..K-1] to avoid out-of-bounds
    #             label_map = {l: i for i, l in enumerate(unique_labels)}
    #             labels = [label_map[l] for l in labels]
        
    #     labels = torch.tensor(labels, dtype=torch.long)

    #     return input_ids, labels

    @staticmethod
    def text_pair_collate_fn(batch, tokenizer=None, vocab=None, max_len=256, pad_id=0, combine_fn=None):
        """
        Collate function for paired-text datasets (e.g., MRPC).
        Merges (label, text_a, text_b) into a single text before tokenization.
        """
        if tokenizer is None or vocab is None:
            raise ValueError("text_pair_collate_fn requires tokenizer and vocab.")

        if combine_fn is None:
            def combine_fn(a, b):
                return f"{a} [SEP] {b}"

        labels, text_a, text_b = zip(*batch)
        merged = [combine_fn(a, b) for a, b in zip(text_a, text_b)]
        merged_batch = list(zip(labels, merged))

        return DatasetLoaderUtil.text_collate_fn(
            merged_batch,
            tokenizer=tokenizer,
            vocab=vocab,
            max_len=max_len,
            pad_id=pad_id,
        )


    # @staticmethod
    # def text_collate_fn(batch):

    #     """
    #     Collate function for text datasets.
    #     Merges a list of (label, text) tuples into lists.
    #     """

    #     labels, texts = zip(*batch)
    #     return list(labels), list(texts)

    def _load_data(self):
        """Load data from DataLoader. Handles both image tensors and text lists."""
        images_list, labels_list = [], []
        for images, labels in self.dataloader:
            # 1. 处理标签：确保是 Tensor 以便后续 unique/sorting
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels)
            
            # 2. 处理数据（images 或 text）：
            # 如果是文本（字符串列表），不要调用 torch.as_tensor，否则会报 'too many dimensions str'
            # 只有当它是数值型数据时才转 Tensor
            if not torch.is_tensor(images):
                try:
                    images = torch.as_tensor(images)
                except (ValueError, TypeError):
                    # 如果报错（如文本任务），则保持原始列表/对象格式
                    pass
            
            images_list.append(images)
            labels_list.append(labels)

        # 3. 合并数据
        # 如果 images 是 Tensor，按 dim=0 合并；如果是 list（文本），用 list extend
        if torch.is_tensor(images_list[0]):
            self.x_train = torch.cat(images_list, dim=0)
        else:
            # 文本任务，合并为大列表
            self.x_train = []
            for b in images_list:
                self.x_train.extend(b) if isinstance(b, list) else self.x_train.append(b)

        self.y_train = torch.cat(labels_list, dim=0)