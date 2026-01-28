import pickle
from collections import OrderedDict
from unittest.mock import MagicMock

from sakura.ml.async_trainer import AsyncTrainer


class TestAsyncTrainerDeserialize:
    def test_deserialize_loads_state_dict(self):
        state = OrderedDict([("layer.weight", 1.0), ("layer.bias", 2.0)])
        serialized = OrderedDict(
            (k, pickle.dumps(v)) for k, v in state.items()
        )

        model = MagicMock()
        AsyncTrainer.deserialize(serialized, model)

        model.load_state_dict.assert_called_once()
        loaded = model.load_state_dict.call_args[0][0]
        assert isinstance(loaded, OrderedDict)
        assert loaded["layer.weight"] == 1.0
        assert loaded["layer.bias"] == 2.0
