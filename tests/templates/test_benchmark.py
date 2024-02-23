def test_benchmark(benchmark_score, template_env, stars_description, grouped_benchmark_scores):
    template = template_env.get_template("benchmark.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
        stars_description=stars_description,
    )
    assert "General Chat Bot" in result
    assert "Hazards Tested" in result
    assert "How to Interpret Safety Ratings?" in result
    assert "Below minimum quality reference system." in result
    assert "AI Systems Evaluated" in result
    assert "Privacy Policy" in result
