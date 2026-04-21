"""End-to-end smoke test for SakuraHFCallback with a real Trainer."""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from sakura.huggingface import SakuraHFCallback


def _tiny_bert_config():
    return transformers.BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, sample_count: int) -> None:
        self._rows = []
        for index in range(sample_count):
            label = index % 2
            input_ids = torch.tensor(
                [1, 2, 3, label + 4, 0, 0, 0, 0], dtype=torch.long
            )
            attention_mask = (input_ids != 0).long()
            self._rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": torch.tensor(label, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        return {key: value.clone() for key, value in row.items()}


def _hf_model_factory():
    return transformers.BertForSequenceClassification(_tiny_bert_config())


def _hf_eval_fn(model, payload) -> dict:
    with torch.no_grad():
        logits = model(
            input_ids=payload["input_ids"],
            attention_mask=payload["attention_mask"],
        ).logits
        predictions = logits.argmax(dim=-1)
    labels = payload["labels"]
    accuracy = float((predictions == labels).float().mean().item())
    return {"val_acc": accuracy}


class TestSakuraHFTrainerEndToEnd:
    def test_real_trainer_run_records_eval_metrics(self, tmp_path):
        torch.manual_seed(0)
        train_dataset = _TinyDataset(sample_count=8)
        eval_dataset = _TinyDataset(sample_count=4)
        eval_payload = {
            "input_ids": torch.stack(
                [eval_dataset[index]["input_ids"] for index in range(len(eval_dataset))]
            ),
            "attention_mask": torch.stack(
                [
                    eval_dataset[index]["attention_mask"]
                    for index in range(len(eval_dataset))
                ]
            ),
            "labels": torch.stack(
                [eval_dataset[index]["labels"] for index in range(len(eval_dataset))]
            ),
        }

        callback = SakuraHFCallback(
            model_factory=_hf_model_factory,
            eval_fn=_hf_eval_fn,
            eval_payload=eval_payload,
            cache_key=None,
            verbose=False,
        )
        model = _hf_model_factory()
        trainer = transformers.Trainer(
            model=model,
            args=transformers.TrainingArguments(
                output_dir=str(tmp_path / "hf-smoke"),
                num_train_epochs=1,
                per_device_train_batch_size=2,
                eval_strategy="no",
                save_strategy="no",
                logging_strategy="no",
                report_to=[],
                disable_tqdm=True,
                seed=0,
                dataloader_num_workers=0,
            ),
            train_dataset=train_dataset,
            callbacks=[callback],
        )

        trainer.train()

        assert len(callback.history) == 1
        assert callback.history[0]["epoch"] == 1
        assert 0.0 <= callback.history[0]["val_acc"] <= 1.0
        assert "elapsed_secs" in callback.history[0]
