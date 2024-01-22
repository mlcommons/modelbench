from helm.benchmark.run import main
import helm.benchmark.run_specs

helm.benchmark.run_specs.INCLUDE_GENERATIVE_HARMS_METRICS = True
if __name__ == "__main__":
    main()
