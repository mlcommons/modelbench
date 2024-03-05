def test_benchmarks(benchmark_score, template_env, grouped_benchmark_scores):
    template = template_env.get_template("benchmarks.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
    )
    assert "AI Safety Benchmarks" in result
    assert "Benchmark for general purpose AI chat model" in result
