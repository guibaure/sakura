import pickle
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from sakura.ml.sakura_trainer import SakuraTrainer
from sakura.ml.epoch.range import range as EpochRange


def _make_trainer(**overrides):
    defaults = dict(
        model=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        metrics=MagicMock(),
        epochs=10,
        model_path="/tmp/model.pth",
        checkpoint_path="/tmp/checkpoint.pth",
    )
    defaults.update(overrides)
    return SakuraTrainer(**defaults)


class TestSakuraTrainer:
    def test_init(self):
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        metrics = MagicMock()
        t = SakuraTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            epochs=50,
            model_path="/tmp/m.pth",
            checkpoint_path="/tmp/c.pth",
            device="cuda",
            device_test="cuda:1",
        )
        assert t._model is model
        assert t._optimizer is optimizer
        assert t._scheduler is scheduler
        assert t._metrics is metrics
        assert t._model_path == "/tmp/m.pth"
        assert t._checkpoint_path == "/tmp/c.pth"
        assert t._device == "cuda"
        assert t._device_test == "cuda:1"

    def test_epochs_range(self):
        t = _make_trainer(epochs=10)
        assert isinstance(t._epochs, EpochRange)

    def test_run_not_implemented(self):
        t = _make_trainer()
        with pytest.raises(NotImplementedError):
            t.run(train_loader=None, test_loader=None)

    def test_train_not_implemented(self):
        t = _make_trainer()
        with pytest.raises(NotImplementedError):
            t.train(loader=None)

    def test_test_not_implemented(self):
        t = _make_trainer()
        with pytest.raises(NotImplementedError):
            t.test(loader=None)

    def test_serialized_state_dict(self):
        state = OrderedDict([("weight", 42), ("bias", 7)])
        model = MagicMock()
        model.cpu.return_value = model
        model.state_dict.return_value = state

        t = _make_trainer(model=model)
        sd = t.serialized_state_dict()

        assert isinstance(sd, OrderedDict)
        for key in state:
            assert key in sd
            assert pickle.loads(sd[key]) == state[key]
