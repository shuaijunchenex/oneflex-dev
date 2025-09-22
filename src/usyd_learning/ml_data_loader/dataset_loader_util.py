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
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels


    # @staticmethod
    # def text_collate_fn(batch):

    #     """
    #     Collate function for text datasets.
    #     Merges a list of (label, text) tuples into lists.
    #     """

    #     labels, texts = zip(*batch)
    #     return list(labels), list(texts)