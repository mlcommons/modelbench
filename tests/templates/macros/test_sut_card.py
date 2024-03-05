def test_display_sut(benchmark_score, template_env):
    template = template_env.get_template("macros/sut_card.html")
    result = template.module.sut_card(benchmark_score)
    assert 'Model performance can be compared to a "moderate quality"' in result
    assert "66% safe responses" in result
    assert "What does 'Fair' mean?" in result
