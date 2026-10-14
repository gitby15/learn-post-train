import torch

from datasets import load_dataset

from transformers import (
    Trainer,
    TrainingArguments,
)

from learn_post_train.model.bert import load_model
from learn_post_train.model.tokenizer import load_tokenizer

from learn_post_train.dataset.civil_comment import (
    CivilDecisionDataset,
)

from learn_post_train.dataset.data_collator import (
    MultiDecisionCollator,
)


def train():
    tokenizer = load_tokenizer()

    train_dataset = CivilDecisionDataset('train')
    eval_dataset=CivilDecisionDataset('test')

    model = load_model()

    training_args = TrainingArguments(
        output_dir="./checkpoints/multi-decision",

        # Batch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,

        # Optimization
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        warmup_steps=0.05,

        # Precision
        bf16=torch.cuda.is_available()
             and torch.cuda.is_bf16_supported(),

        # Logging
        logging_steps=100,

        # Evaluation
        eval_strategy="steps",
        eval_steps=200,

        # Checkpoints
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        # Misc
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultiDecisionCollator(tokenizer),
    )

    trainer.train()

    trainer.save_model("./checkpoints/multi-decision-final")

    tokenizer.save_pretrained(
        "./checkpoints/multi-decision-final"
    )


if __name__ == "__main__":
    train()