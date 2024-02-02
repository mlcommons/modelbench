def test_display_breadcrumb(benchmark_score, template_env):
    template = template_env.get_template("macros/breadcrumb.html")
    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition)
    assert "Benchmarks" in result
    assert "General Chat Bot" in result
    assert 'href="general_chat_bot_benchmark.html"' in result
    result = template.module.breadcrumb(None, benchmark_score.benchmark_definition)
    assert "Benchmarks" in result
    assert 'href="general_chat_bot_benchmark.html"' not in result
    assert "General Chat Bot" in result
