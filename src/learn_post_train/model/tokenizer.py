from learn_post_train.model.const import MODEL_ID
from transformers import AutoTokenizer
_tokenizer = None


def load_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return _tokenizer

