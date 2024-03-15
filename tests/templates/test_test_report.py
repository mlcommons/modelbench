import re
from bs4 import BeautifulSoup


def test_test_report(benchmark_score, template_env):
    template = template_env.get_template("test_report.html")
    result = template.render(
        benchmark_score=benchmark_score,
    )

    html = BeautifulSoup(result, "html.parser")
    # TODO(mkly)
    # assert html.find(string=re.compile("__test__.stars.3.explanation"))
    assert html.find(string=re.compile("__test__.general_chat_bot_benchmark.name"))
    assert html.find(string=re.compile("Test Report"))
    assert html.find(string=re.compile("Tests Run"))
    assert html.find(string=re.compile("Test Details"))
    assert html.find(string=re.compile("Don't see the tests you are looking for?"))
