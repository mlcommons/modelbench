def test_use_hazards_limitations(benchmark_score, template_env):
    template = template_env.get_template("macros/use_hazards_limitations.html")
    result = template.module.use_hazards_limitations(benchmark_score.benchmark_definition)
    assert "Use Case" in result
    assert "Lorem ipsum dolor sit amet" in result
    assert "Hazards Tested" in result
    assert "Limitations" in result
