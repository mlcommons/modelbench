from coffee.helm import HelmSut
from coffee.benchmark import RidiculousBenchmark


def test_class_basics():
    assert RidiculousBenchmark.name() == "Ridiculous Benchmark"
    assert RidiculousBenchmark.path_name() == "ridiculous_benchmark"


def test_instance_basics():
    rb = RidiculousBenchmark(HelmSut.GPT2, {})
    assert rb.name() == "Ridiculous Benchmark"
    assert rb.path_name() == "ridiculous_benchmark"


# not testing bechmark scoring for the moment because it's all throwaway
