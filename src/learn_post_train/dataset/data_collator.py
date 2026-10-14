import torch
from torch.nn.utils.rnn import pad_sequence


class MultiDecisionCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        )

        decision_positions = pad_sequence(
            [f["decision_positions"] for f in features],
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=0.0,
        )

        question_mask = pad_sequence(
            [
                torch.ones(
                    len(f["labels"]),
                    dtype=torch.float32,
                )
                for f in features
            ],
            batch_first=True,
            padding_value=0.0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decision_positions": decision_positions,
            "labels": labels,
            "question_mask": question_mask,
        }