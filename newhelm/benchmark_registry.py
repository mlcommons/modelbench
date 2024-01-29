from newhelm.instance_factory import InstanceFactory
from newhelm.benchmark import BaseBenchmark

# The list of all Benchmarks with assigned names.
BENCHMARKS = InstanceFactory[BaseBenchmark]()
