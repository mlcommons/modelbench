from coffee.helm import HelmSut, BbqHelmTest
from coffee.benchmark import MakeshiftBiasBenchmark


def test_class_basics():
    assert MakeshiftBiasBenchmark.name() == "Makeshift Bias Benchmark"
    assert MakeshiftBiasBenchmark.path_name() == "makeshift_bias_benchmark"
    assert MakeshiftBiasBenchmark.tests()[0].__class__ == BbqHelmTest


def test_instance_basics():
    rb = MakeshiftBiasBenchmark(HelmSut.GPT2, {})
    assert rb.sut == HelmSut.GPT2


# not testing bechmark scoring for the moment because it's all throwaway
