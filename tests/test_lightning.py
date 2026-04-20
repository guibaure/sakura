"""Smoke test for the Lightning integration."""

from __future__ import annotations

import pytest

lightning = pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

from sakura.lightning import SakuraTrainer


class _TinyModule(lightning.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        logits = self(x)
        return torch.nn.functional.cross_entropy(logits, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _make_loader(sample_count: int) -> DataLoader:
    features = torch.randn(sample_count, 4)
    targets = torch.randint(0, 2, (sample_count,))
    return DataLoader(TensorDataset(features, targets), batch_size=4)


class TestSakuraLightningTrainer:
    def test_run_records_async_validation_history(self):
        """Training should complete and record standalone validation metrics."""
        torch.manual_seed(0)
        train_loader = _make_loader(8)
        val_loader = _make_loader(8)

        trainer = SakuraTrainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            model_factory=_TinyModule,
            val_loader_factory=lambda: val_loader,
            verbose=False,
        )

        model = _TinyModule()
        trained = trainer.run(model, train_loader)

        assert trained is model
        assert len(trainer.history) == 1
        assert trainer.history[0]["worker_name"] == "<standalone>"
        assert trainer.history[0]["val_loss"] >= 0
        assert trainer.best_val_loss == trainer.history[0]["val_loss"]
