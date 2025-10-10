
class TokenizerFactory:
    @staticmethod
    def create_tokenizer(tokenizer_type, tokenizer_args):
        if tokenizer_type == "simple":
            from usyd_learning.ml_algorithms.tokenizer.simple_tokenizer import SimpleTokenizer
            return SimpleTokenizer(tokenizer_args)
        elif tokenizer_type == "bert":
            from usyd_learning.ml_algorithms.tokenizer.bert_tokenizer import BertTokenizer
            return BertTokenizer(tokenizer_args)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")