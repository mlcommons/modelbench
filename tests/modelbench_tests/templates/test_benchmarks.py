import re
from bs4 import BeautifulSoup


def test_benchmarks_mlc(benchmark_score, template_env_mlc, grouped_benchmark_scores):
    template = template_env_mlc.get_template("benchmarks.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
        mlc_branding=True,
    )
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("AI Safety Benchmarks"))
    assert html.find(string=re.compile("__test__.general_purpose_ai_chat_benchmark_0_5.name"))
    assert html.find(string=re.compile("Don't see the benchmark you are looking for?"))


def test_benchmarks_generic(benchmark_score, template_env, grouped_benchmark_scores):
    template = template_env.get_template("benchmarks.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
    )
    html = BeautifulSoup(result, "html.parser")
    assert not html.find(string=re.compile("Don't see the benchmark you are looking for?"))
