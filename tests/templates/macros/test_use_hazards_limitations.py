import re
from bs4 import BeautifulSoup


def test_use_hazards_limitations(benchmark_score, template_env):
    template = template_env.get_template("macros/use_hazards_limitations.html")
    result = template.module.use_hazards_limitations(benchmark_score.benchmark_definition)
    html = BeautifulSoup(result, "html.parser")
    assert html.find(string=re.compile("Use Case"))
    assert html.find(string=re.compile("__test__.general_purpose_ai_chat_benchmark.application"))
    assert html.find(string=re.compile("Hazards Tested"))
    assert html.find(string=re.compile("Limitations"))
