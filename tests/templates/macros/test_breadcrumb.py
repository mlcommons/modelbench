import re


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


def test_breadcrumb_no_link_benchmarks_page_type(benchmark_score, template_env):
    template = template_env.get_template("macros/breadcrumb.html")
    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition, "benchmarks")
    assert "Benchmarks" in result
    assert re.search("<li.*>Benchmarks</li>", result) is not None
    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition)
    assert re.search("<li.*>Benchmarks</li>", result) is None
    assert re.search(r"<li.*><a.*>Benchmarks</a></li>", result) is not None
