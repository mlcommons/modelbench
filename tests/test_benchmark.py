from coffee.helm import HelmSut, BbqHelmTest
from coffee.benchmark import MakeshiftBiasBenchmark, piecewise_normalization
import pytest
import numpy as np


def test_class_basics():
    assert MakeshiftBiasBenchmark.name() == "Makeshift Bias Benchmark"
    assert MakeshiftBiasBenchmark.path_name() == "makeshift_bias_benchmark"
    assert MakeshiftBiasBenchmark.tests()[0].__class__ == BbqHelmTest


def test_instance_basics():
    rb = MakeshiftBiasBenchmark(HelmSut.GPT2, {})
    assert rb.sut == HelmSut.GPT2


@pytest.mark.parametrize(
    "min_possible,mid_point,max_possible",
    [
        (0, 3, 5),
        (-10, 0, 10),
        (-17, -5, 0),
        (0, 99, 100),
    ],
)
def test_piecewise_normalization(min_possible, mid_point, max_possible):
    normalizer_fn = piecewise_normalization(min_possible, mid_point, max_possible)

    assert normalizer_fn(min_possible) == 0
    assert normalizer_fn(mid_point) == 3
    assert normalizer_fn(max_possible) == 5

    lower_half = (mid_point + min_possible) / 2
    assert 0 <= normalizer_fn(lower_half) <= 3
    upper_half = (max_possible + mid_point) / 2
    assert 3 <= normalizer_fn(upper_half) <= 5

    # Sample 20 points along the function to ensure they are monotonic.
    xs = np.linspace(min_possible, max_possible, num=20)
    for i in range(len(xs) - 1):
        lower = normalizer_fn(xs[i])
        higher = normalizer_fn(xs[i + 1])
        assert lower < higher, f"Expected f({xs[i]}) to be less than f({xs[i+1]})"


def test_piecewise_normalization_midpoint_low():
    with pytest.raises(AssertionError) as err_info:
        piecewise_normalization(5, 5, 10)
    assert "If mid_point is <= min_possible, nothing can score < 3." in str(err_info)


def test_piecewise_normalization_midpoint_high():
    with pytest.raises(AssertionError) as err_info:
        piecewise_normalization(5, 10, 10)
    assert "If mid_point is > max_possible, nothing can score > 3." in str(err_info)


# not testing bechmark scoring for the moment because it's all throwaway
