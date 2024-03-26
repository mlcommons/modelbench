import re
from bs4 import BeautifulSoup


def test_benchmark(benchmark_score, template_env, grouped_benchmark_scores):
    template = template_env.get_template("benchmark.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
    )
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("__test__.general_chat_bot_benchmark.name"))
    assert html.find(string=re.compile("__test__.grades.3.long_description"))
    assert html.find(string=re.compile("Hazards Tested"))
    assert html.find(string=re.compile("How to Interpret Safety Ratings?"))
    assert html.find(string=re.compile("AI Systems Evaluated"))


def test_benchmark_container_embed_class(benchmark_score, template_env_view_embed, grouped_benchmark_scores):
    template = template_env_view_embed.get_template("benchmark.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
        view_embed=True,
    )
    html = BeautifulSoup(result, "html.parser")
    assert html.find("div", {"class": "mlc--container__embed"})
