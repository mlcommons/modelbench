def test_test_report(benchmark_score, template_env, stars_description):
    template = template_env.get_template("test_report.html")
    result = template.render(
        benchmark_score=benchmark_score,
        stars_description=stars_description,
    )
    assert "Test Report" in result
    assert "66% safe responses" in result
    assert "Tests Run" in result
    assert "Test Details" in result
    assert "General Chat Bot" in result
    assert "Don't see the tests you are looking for?" in result
    assert "Privacy Policy" in result
