import pytest
from src.evaluation.benchmark_inference import summarize


def test_latency_is_total_time_throughput_not_mean_instantaneous_fps():
    result = summarize([10, 20, 30])
    assert result["mean_ms"] == 20
    assert result["median_ms"] == 20
    assert result["p95_ms"] == 29
    assert result["fps_from_total_time"] == 50


@pytest.mark.parametrize("samples", [[], [0], [float("nan")], [-1]])
def test_invalid_timings_rejected(samples):
    with pytest.raises(ValueError):
        summarize(samples)
