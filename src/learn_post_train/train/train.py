import torch
from transformers import (
    Trainer,
    TrainingArguments,
)

from learn_post_train.dataset.civil_comment import (
    CivilCommentCollector,
    CivilCommentDataset,
)
from learn_post_train.model.bert import load_model
from learn_post_train.model.tokenizer import load_tokenizer


def train():
    tokenizer = load_tokenizer()
    train_dataset = CivilCommentDataset('train')
    eval_dataset=CivilCommentDataset('validation')
    model = load_model()

    is_bf_16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print('is_bf_16 ========== ', is_bf_16)
    training_args = TrainingArguments(
        output_dir="./checkpoints/multi-decision",

        # Batch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        # Optimization
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=1,
        warmup_steps=0.05,
        optim=(
            "adamw_torch_fused"
            if torch.cuda.is_available()
            else "adamw_torch"
        ),

        # Operator fusion
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="default",

        # Precision
        bf16=is_bf_16,

        # Logging
        logging_steps=500,

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,

        # Checkpoints
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,

        # Misc
        remove_unused_columns=False,
        train_sampling_strategy="sequential",
        report_to="tensorboard",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CivilCommentCollector(tokenizer),
    )

    trainer.train()

    trainer.save_model("./checkpoints/multi-decision-final")

    tokenizer.save_pretrained(
        "./checkpoints/multi-decision-final"
    )


if __name__ == "__main__":
    train()