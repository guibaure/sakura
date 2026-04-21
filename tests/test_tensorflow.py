"""Tests for the TensorFlow / Keras integration without requiring TensorFlow."""

from __future__ import annotations

import importlib
import sys
import types


class _FakeKerasModel:
    def __init__(self) -> None:
        self._weights = [0.0, 0.0]

    def get_weights(self):
        return list(self._weights)

    def set_weights(self, weights):
        self._weights = list(weights)


def _fake_model_factory():
    return _FakeKerasModel()


def _fake_val_fn(model, payload) -> dict:
    offset = payload
    total = sum(model.get_weights())
    return {
        "val_loss": float(total + offset),
        "val_acc": float(total / 10.0),
    }


def _load_tensorflow_module(monkeypatch):
    fake_tensorflow = types.ModuleType("tensorflow")
    fake_keras = types.ModuleType("tensorflow.keras")
    fake_callbacks = types.ModuleType("tensorflow.keras.callbacks")

    class _FakeCallback:
        def __init__(self) -> None:
            self.model = None

    fake_callbacks.Callback = _FakeCallback
    fake_keras.callbacks = fake_callbacks
    fake_tensorflow.keras = fake_keras

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", fake_keras)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.callbacks", fake_callbacks)
    sys.modules.pop("sakura.tensorflow", None)
    return importlib.import_module("sakura.tensorflow")


class TestSakuraKerasCallback:
    def test_dispatches_and_records_metrics(self, monkeypatch):
        module = _load_tensorflow_module(monkeypatch)
        callback = module.SakuraKerasCallback(
            model_factory=_fake_model_factory,
            val_fn=_fake_val_fn,
            val_payload=1.0,
            verbose=False,
        )
        callback.model = _FakeKerasModel()

        callback.model.set_weights([2.0, 4.0])
        callback.on_epoch_end(0)

        callback.model.set_weights([1.0, 1.0])
        callback.on_epoch_end(1)
        callback.on_train_end()

        assert len(callback.history) == 2
        assert callback.history[0]["epoch"] == 0
        assert callback.history[0]["val_loss"] == 7.0
        assert callback.history[0]["val_acc"] == 0.6
        assert callback.history[1]["epoch"] == 1
        assert callback.history[1]["val_loss"] == 3.0
        assert callback.history[1]["val_acc"] == 0.2
        for row in callback.history:
            assert "elapsed_secs" in row
