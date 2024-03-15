import re
from bs4 import BeautifulSoup


def test_interpret_safety_ratings(template_env):
    template = template_env.get_template("macros/interpret_safety_ratings.html")
    result = template.module.interpret_safety_ratings()
    html = BeautifulSoup(result, "html.parser")

    assert html.find(string=re.compile("How to Interpret Safety Ratings?"))
    assert html.find(string=re.compile("__test__.general.interpret_safety_ratings"))
    # TODO(mkly)
    # assert html.find(string=re.compile("__test__.stars.3.short_description"))
    # assert html.find(string=re.compile("__test__.stars.1.short_description"))
    # assert html.find(string=re.compile("__test__.stars.5.rank"))
