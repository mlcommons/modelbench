def test_display_benchmark_card(benchmark_score, template_env):
    template = template_env.get_template("macros/benchmark_card.html")
    result = template.module.benchmark_card(True, benchmark_score.benchmark_definition)
    assert "Benchmark for general purpose AI chat model" in result
    assert "An open benchmark for evaluating the safety of AI models" in result
    result_no_header = template.module.benchmark_card(False, benchmark_score.benchmark_definition)
    assert "Benchmark for general purpose AI chat model" not in result_no_header
    assert "An open benchmark for evaluating the safety of AI models" in result
