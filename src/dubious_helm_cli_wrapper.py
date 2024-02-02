import helm.benchmark.run_specs
from helm.benchmark.run import main

helm.benchmark.run_specs.INCLUDE_GENERATIVE_HARMS_METRICS = True
if __name__ == "__main__":
    main()
