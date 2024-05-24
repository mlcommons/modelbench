import re
from bs4 import BeautifulSoup


def test_display_breadcrumb(benchmark_score, template_env):
    template = template_env.get_template("macros/breadcrumb.html")
    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition)
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("Benchmarks"))
    assert html.find(string=re.compile("__test__.general_purpose_ai_chat_benchmark.name"))
    assert html.find("a", href="general_purpose_ai_chat_benchmark.html")

    result = template.module.breadcrumb(None, benchmark_score.benchmark_definition)
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("Benchmarks"))
    assert not html.find("a", href="general_purpose_ai_chat_benchmark.html")
    assert html.find(string=re.compile("__test__.general_purpose_ai_chat_benchmark.name"))


def test_breadcrumb_no_link_benchmarks_page_type(benchmark_score, template_env):
    template = template_env.get_template("macros/breadcrumb.html")
    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition, "benchmarks")
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("Benchmarks"))
    assert not html.find("a", string=re.compile("Benchmarks"))

    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition)
    html = BeautifulSoup(result, "html.parser")
    assert html.find("a", string=re.compile("Benchmarks"))


def test_breadcrumb_generic_arg(benchmark_score, template_env):
    template = template_env.get_template("macros/breadcrumb.html")
    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition)
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("MLCommons"))

    result = template.module.breadcrumb(benchmark_score, benchmark_score.benchmark_definition, generic=True)
    html = BeautifulSoup(result, "html.parser")
    assert not html.find(string=re.compile("MLCommons"))
    assert html.find(string=re.compile("Home"))
