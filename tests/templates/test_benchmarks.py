import re
from bs4 import BeautifulSoup


def test_benchmarks(benchmark_score, template_env, grouped_benchmark_scores):
    template = template_env.get_template("benchmarks.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
    )
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("AI Safety Benchmarks"))
    assert html.find(string=re.compile("__test__.general_chat_bot_benchmark.name"))
