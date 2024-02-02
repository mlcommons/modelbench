def test_use_harms_limitations(benchmark_score, template_env):
    template = template_env.get_template("macros/use_harms_limitations.html")
    result = template.module.use_harms_limitations(benchmark_score.benchmark_definition)
    assert "Use Case" in result
    assert "Lorem ipsum dolor sit amet" in result
    assert "Harms Tested" in result
    assert "Limitations" in result
