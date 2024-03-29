import re
from bs4 import BeautifulSoup


def test_display_sut(benchmark_score, template_env):
    template = template_env.get_template("macros/sut_card.html")
    result = template.module.sut_card(benchmark_score)
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("__test__.grades.1.long_description"))
    assert html.find(string=re.compile("__test__.grades.1.explanation"))
    assert html.find(string=re.compile("__test__.grades.1.rank"))
