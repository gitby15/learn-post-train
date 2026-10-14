from learn_post_train.model.const import MODEL_ID
from transformers import AutoTokenizer
_tokenizer = None




def load_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    # 决策模型需要增加的special token
    special_tokens = [
        "[MSG]",
        "[QUESTION]",
        "[DECISION]",
    ]
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })
    
    return _tokenizer


def build_input(msg: str, questions: list[str]):
    text = f"[STATE] {msg}"
    for question in questions:
        text += f" [QUESTION] {question} [DECISION]"
    tokenizer = load_tokenizer()
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_tensors="pt",
    )
    decision_token_id = tokenizer.convert_tokens_to_ids("[DECISION]")
    positions = (
        encoded["input_ids"][0] == decision_token_id
    ).nonzero(as_tuple=True)[0]
    return encoded, positions




def tokenize(text: str):
    tokenizer = load_tokenizer()
    inputs = tokenizer(text, return_tensors="pt")
    return inputs

def detokenize(logits):
    tokenizer = load_tokenizer()
    predicted_token = tokenizer.decode(logits)
    return predicted_token