from sakura.functional import defaultMetrics, asr_metrics


class TestDefaultMetrics:
    def test_default_metrics_structure(self):
        assert hasattr(defaultMetrics, "train")
        assert hasattr(defaultMetrics, "test")
        assert "current" in defaultMetrics.train
        assert "best" in defaultMetrics.train
        assert "current" in defaultMetrics.test
        assert "best" in defaultMetrics.test

    def test_default_metrics_values(self):
        assert defaultMetrics.train["current"]["loss"] == 0
        assert defaultMetrics.train["current"]["accuracy"] == 0
        assert defaultMetrics.train["best"]["loss"] == 0
        assert defaultMetrics.train["best"]["accuracy"] == 0


class TestAsrMetrics:
    def test_asr_metrics_structure(self):
        assert hasattr(asr_metrics, "train")
        assert hasattr(asr_metrics, "test")
        assert "current" in asr_metrics.train
        assert "best" in asr_metrics.train

    def test_asr_metrics_values(self):
        assert asr_metrics.train["current"]["loss"] == 0
        assert asr_metrics.train["current"]["wer"] == 100
        assert asr_metrics.train["current"]["cer"] == 100
