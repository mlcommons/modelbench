import re
from bs4 import BeautifulSoup


def test_benchmark_mlc(benchmark_score, template_env_mlc, grouped_benchmark_scores):
    template = template_env_mlc.get_template("benchmark.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
        mlc_branding=True,
    )
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("__test__.general_purpose_ai_chat_benchmark.name"))
    assert html.find(string=re.compile("__test__.grades.1.long_description"))
    assert html.find(string=re.compile("Hazards Tested"))
    assert html.find(string=re.compile("How to Interpret Safety Ratings?"))
    assert html.find(string=re.compile("AI Systems Evaluated"))
    assert html.find(string=re.compile("Don't see the AI system you are looking for?"))


def test_benchmark_generic(benchmark_score, template_env, grouped_benchmark_scores):
    template = template_env.get_template("benchmark.html")
    result = template.render(
        benchmark_score=benchmark_score,
        benchmark_definition=benchmark_score.benchmark_definition,
        grouped_benchmark_scores=grouped_benchmark_scores,
    )
    html = BeautifulSoup(result, "html.parser")
    assert not html.find(string=re.compile("AI Systems Evaluated"))
    assert not html.find(string=re.compile("Don't see the AI system you are looking for?"))


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
