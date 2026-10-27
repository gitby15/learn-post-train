
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from learn_post_train.model.const import MODEL_ID
from learn_post_train.model.tokenizer import (
    load_tokenizer,
)

'''
示例输入
result = model.decide(
    state="You are an idiot.",
    questions={
        "toxic": {"type": "noul", "instructions":"Is this a toxic message?"},
        "is_insult": {"type": "noul", "instructions":"Is this comment an insult?"},
        "is_threat": {"type": "noul", "instructions":"Is this comment a threat?"},
    },
)

# 示例输出
{
    "toxic": {
        "noul": 0.965,
        "confidence": 0.88,
        "answer_confidence": 0.85,
        "probabilities": [0.965, 0.035],
    },
    "is_insult": {
        "noul": 0.965,
        "confidence": 0.88,
        "answer_confidence": 0.85,
        "probabilities": [0.965, 0.035],
    },
    "is_threat": {
        "noul": 0.965,
        "confidence": 0.88,
        "answer_confidence": 0.85,
        "probabilities": [0.965, 0.035],
    },
}
'''

'''
模型在实际处理的时候，具体做法是，state + question拆分成多个pair
有几个问题，就产生几个pair，将他们组成batch来批处理
# 输入格式：
[CLS]
You are an idiot.
[SEP]
Does this comment contain an insult?
[SEP]
'''
_model = None
class MultiDecisionModel(nn.Module):
    def __init__(self, model_name, vocab_size):
        super().__init__()

        # 只加载 ModernBERT Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(vocab_size)
        hidden_size = self.encoder.config.hidden_size

        # [CLS] 已经聚合了整个 state-question pair 的 self-attention 信息。
        # 所有问题共享同一个分类头，输出未经过 sigmoid 的 Yes logit。
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels = None,
    ):
        # input_ids、attention_mask: [Q, L]
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # encoder的输出，形状是[Q, L, H]
        # [Q, L, H] ==> [Q, H] => [Q, 1]
        last_hidden_state = outputs.last_hidden_state
        # 最后一层Transformer的输出，用CLS来做池化
        hidden_state = last_hidden_state[:, 0, :]
        logits = self.classifier(hidden_state).squeeze(-1)

        loss = None

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)



        return {"logits": logits, "loss": loss}


        



def load_model():
    global _model
    if _model is not None:
        return _model

    tokenizer = load_tokenizer()
    model = MultiDecisionModel(
        model_name=MODEL_ID,
        vocab_size=len(tokenizer),
    )
    
    return model

