import re
from bs4 import BeautifulSoup


def test_interpret_safety_ratings(template_env):
    template = template_env.get_template("macros/interpret_safety_ratings.html")
    result = template.module.interpret_safety_ratings()
    html = BeautifulSoup(result, "html.parser")

    assert html.find(string=re.compile("How to Interpret Safety Ratings?"))
    assert html.find(string=re.compile("__test__.general.interpret_safety_ratings"))
    assert html.find(string=re.compile("__test__.grades.3.long_description"))
    assert html.find(string=re.compile("__test__.grades.1.explanation"))
    assert html.find(string=re.compile("__test__.grades.4.rank"))
