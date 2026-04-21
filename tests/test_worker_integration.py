"""Worker-backed integration test for the Zakuro transport path."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from zakuro.worker.runner import Worker

from sakura.ml.async_trainer import AsyncTrainer


class _Metrics:
    def __init__(self) -> None:
        self.test: dict = {}


class _FakeTrainer:
    def __init__(self, epochs: int) -> None:
        self._epochs = list(range(epochs))
        self._epoch = 0
        self._metrics = _Metrics()
        self._value = 40

    def serialized_state_dict(self) -> dict:
        return {"value": self._value}

    def train(self, train_loader=None):  # noqa: ARG002
        self._value += 1


class _WorkerModel:
    def __init__(self) -> None:
        self.state_dict_data: dict = {}

    def load_state_dict(self, state_dict: dict) -> None:
        self.state_dict_data = dict(state_dict)


def _worker_model_factory():
    return _WorkerModel()


def _worker_test_fn(model) -> dict:
    return {
        "value": model.state_dict_data["value"],
        "worker_pid": os.getpid(),
    }


class TestWorkerBackedAsyncTrainer:
    def test_dispatches_to_real_http_worker(self):
        trainer = _FakeTrainer(epochs=2)

        try:
            worker = Worker.spawn(name="sakura-pytest-worker", transport="http")
        except PermissionError as exc:
            pytest.skip(f"worker integration requires local socket permissions: {exc}")

        with worker:
            async_trainer = AsyncTrainer(
                trainer=trainer,
                model_factory=_worker_model_factory,
                test_fn=_worker_test_fn,
                val_compute=worker.compute(),
            )
            async_trainer.run()

        assert trainer._metrics.test["value"] == 42
        assert trainer._metrics.test["worker_pid"] != os.getpid()
