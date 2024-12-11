import re
from bs4 import BeautifulSoup


def test_display_benchmark_card(benchmark_score, benchmark_uid, template_env):
    template = template_env.get_template("macros/benchmark_card.html")
    result = template.module.benchmark_card(True, benchmark_score.benchmark_definition)
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile(f"__test__.{benchmark_uid}.name"))
    assert html.find(string=re.compile(f"__test__.{benchmark_uid}.tagline"))

    result_no_header = template.module.benchmark_card(False, benchmark_score.benchmark_definition)
    html_no_header = BeautifulSoup(result_no_header, "html.parser")
    assert not html_no_header.find(string=re.compile(f"__test__.{benchmark_uid}.name"))
    assert html_no_header.find(string=re.compile(f"__test__.{benchmark_uid}.tagline"))
