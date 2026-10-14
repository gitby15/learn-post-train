import torch
from torch import nn
from learn_post_train.model.const import MODEL_ID
from learn_post_train.model.tokenizer import build_input, load_tokenizer, tokenize, detokenize
from transformers import AutoModel
import torch.nn.functional as F

'''
示例输入
result = model.decide(
    state="You are an idiot.",
    questions=[
        "Does this comment contain an insult?",
        "Does this comment contain a threat?",
        "Is this comment toxic?",
    ],
)

# 示例输出
# [0.96, 0.03, 0.91]
'''

'''
# 输入格式：
[CLS]
[STATE] You are an idiot.
[QUESTION] Does this comment contain an insult?
[DECISION]
[QUESTION] Does this comment contain a threat?
[DECISION]
[QUESTION] Is this comment toxic?
[DECISION]
[SEP]
'''



_model = None


class MultiDecisionModel(nn.Module):
    def __init__(self, model_name, vocab_size):
        super().__init__()

        # 只加载 ModernBERT Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        # 因为新增了special token, 所以需要resize一下
        self.encoder.resize_token_embeddings(vocab_size)
        hidden_size = self.encoder.config.hidden_size
        # 所有 Decision Token 共享同一个 Head
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        decision_positions,
        labels=None,
        question_mask=None,
    ):
        # [B, L, H]
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        batch_size = hidden_states.shape[0]

        batch_indices = torch.arange(
            batch_size,
            device=hidden_states.device,
        ).unsqueeze(1)

        # [B, Q, H]
        decision_states = hidden_states[
            batch_indices,
            decision_positions,
        ]

        # [B, Q, 1] -> [B, Q]
        logits = self.classifier(
            decision_states
        ).squeeze(-1)


        loss = None
        if labels is not None:
            loss_per_question = F.binary_cross_entropy_with_logits(
                logits.float(),
                labels.float(),
                reduction="none",
            )
            if question_mask is None:
                question_mask = torch.ones_like(labels)
            question_mask = question_mask.float()

            loss = (
                loss_per_question * question_mask
            ).sum() / question_mask.sum().clamp_min(1)


        return {
            'loss': loss,
            'logits': logits
        }



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



def test():
    model = load_model()
    encoded, positions = build_input(
        "You are an idiot.",
        [
            "Is this an insult?",
            "Is this a threat?",
            "Is this toxic?",
            "Is this happy?",
        ],
    )
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    output = model(input_ids, attention_mask, positions)

    print("====")
    print(output['logits'])



if __name__ == "__main__":
    test()