def test_interpret_safety_ratings(template_env):
    template = template_env.get_template("macros/interpret_safety_ratings.html")
    result = template.module.interpret_safety_ratings()
    assert "How to Interpret Safety Ratings?" in result
    assert "Description goes here" in result
    assert "At or above moderate quality reference system." in result
    assert "Below minimum quality reference system." in result
    assert "Excellent" in result
