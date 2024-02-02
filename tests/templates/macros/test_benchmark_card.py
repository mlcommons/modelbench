def test_display_benchmark_card(benchmark_score, template_env):
    template = template_env.get_template("macros/benchmark_card.html")
    result = template.module.benchmark_card(True, benchmark_score.benchmark_definition)
    assert "General Chat Bot Benchmark" in result
    assert "Lorem ipsum dolor sit amet" in result
    result_no_header = template.module.benchmark_card(False, benchmark_score.benchmark_definition)
    assert "General Chat Bot Benchmark" not in result_no_header
    assert "Lorem ipsum dolor sit amet" in result
