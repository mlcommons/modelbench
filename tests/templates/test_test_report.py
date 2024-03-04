def test_test_report(benchmark_score, template_env):
    template = template_env.get_template("test_report.html")
    result = template.render(
        benchmark_score=benchmark_score,
    )
    assert "Test Report" in result
    assert "66% safe responses" in result
    assert "Tests Run" in result
    assert "Test Details" in result
    assert "Benchmark for general purpose AI chat model" in result
    assert "Don't see the tests you are looking for?" in result
