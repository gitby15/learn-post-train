import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from learn_post_train.model.bert import load_model
from learn_post_train.model.tokenizer import load_tokenizer
from learn_post_train.dataset.civil_comment import CivilDecisionDataset
from learn_post_train.dataset.data_collator import MultiDecisionCollator

def smoke_test():

    dataset = CivilDecisionDataset()

    tokenizer = load_tokenizer()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=MultiDecisionCollator(tokenizer),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model().to(device)
    model.train()
    print("start to train")

    batch = next(iter(loader))
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    outputs = model(**batch)

    print("Input:", batch["input_ids"].shape)
    print("Decisions:", outputs["logits"].shape)
    print("Loss:", outputs["loss"].item())

    assert outputs["logits"].shape == batch["labels"].shape
    assert torch.isfinite(outputs["loss"])

    outputs["loss"].backward()

    assert model.classifier.weight.grad is not None

    print("Smoke test passed")


if __name__ == "__main__":
    smoke_test()