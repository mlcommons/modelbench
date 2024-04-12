import re
from bs4 import BeautifulSoup


def test_test_report(benchmark_score, template_env):
    template = template_env.get_template("test_report.html")
    result = template.render(
        benchmark_score=benchmark_score,
    )

    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("__test__.grades.1.explanation"))
    assert html.find(string=re.compile("__test__.general_purpose_ai_chat_benchmark.name"))
    assert html.find(string=re.compile("MLCommons AI Safety v0.5 Benchmark Report"))
    assert html.find(string=re.compile("Hazard Scoring Details"))
    assert html.find(string=re.compile("Run Details"))
    assert html.find(string=re.compile("Don't see the tests you are looking for?"))
