def test_benchmarks(benchmark_score, template_env, stars_description, grouped_benchmark_scores):
    template = template_env.get_template("benchmarks.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
        stars_description=stars_description,
    )
    assert "AIP Safety Benchmarks" in result
    assert "General Chat Bot Benchmark" in result
    assert "Privacy Policy" in result
