from sakura.ml.epoch.range import range as EpochRange


class TestEpochRange:
    def test_init_defaults(self):
        r = EpochRange()
        assert r.start == 1
        assert r.stop == 100
        assert r.total == 99

    def test_init_custom(self):
        r = EpochRange(start=5, stop=20)
        assert r.start == 5
        assert r.stop == 20

    def test_iteration(self):
        r = EpochRange(start=0, stop=5)
        values = list(r)
        assert values == [1, 2, 3, 4]

    def test_stop_iteration(self):
        r = EpochRange(start=0, stop=1)
        # First __next__ increments current to 1, which is not < stop(1),
        # so it raises StopIteration immediately.
        values = list(r)
        assert values == []

    def test_total_property(self):
        r = EpochRange(start=10, stop=50)
        assert r.total == 40

    def test_iter_returns_self(self):
        r = EpochRange()
        assert iter(r) is r

    def test_empty_range(self):
        r = EpochRange(start=5, stop=5)
        values = list(r)
        assert values == []
