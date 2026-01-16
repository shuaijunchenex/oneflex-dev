import torch

class DatasetLoaderUtil:
    """
    " DataLoader Util class
    """
    
    # torchtext datasets

    
    @staticmethod
    def text_collate_fn(batch, tokenizer=None, vocab=None, max_len=256, pad_id=0):
        """
        Collate function for text datasets.
        Converts list of (label, text) into (padded_input_ids, labels).

        Args:
            batch: list of (label, text) pairs
            tokenizer: callable, text -> list of tokens
            vocab: vocab object, supports vocab[token] -> id
            max_len: maximum sequence length (truncate if longer)
            pad_id: index used for padding

        Returns:
            input_ids: LongTensor [B, L]
            labels:    LongTensor [B]
        """
        if tokenizer is None or vocab is None:
            raise ValueError("text_collate_fn requires tokenizer and vocab.")

        labels, texts = zip(*batch)

        # tokenize + map to ids
        tokenized = [tokenizer(t) for t in texts]
        ids = [[vocab[token] for token in toks] for toks in tokenized]

        # pad / truncate
        batch_max_len = min(max(len(seq) for seq in ids), max_len)
        padded = []
        for seq in ids:
            if len(seq) > batch_max_len:
                seq = seq[:batch_max_len]
            else:
                seq = seq + [pad_id] * (batch_max_len - len(seq))
            padded.append(seq)

        input_ids = torch.tensor(padded, dtype=torch.long)
        
        if labels and isinstance(labels[0], str):
            label_map = {l: i for i, l in enumerate(sorted(set(labels)))}
            labels = [label_map[l] for l in labels]
            
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels

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