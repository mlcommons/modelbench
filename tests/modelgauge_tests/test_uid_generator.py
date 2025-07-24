from modelgauge.sut_specification import SUTUIDGenerator, SUTDefinition


def test_uid():
    d = SUTDefinition()
    d.add("model", "chatgpt-4o")
    d.add("maker", "openai")
    d.add("driver", "openai")
    d.add("provider", "openai")
    d.add("temperature", 0.8)
    d.add("top_p", 6)
    d.add("top_k", 5)
    d.add("moderated", True)
    d.add("reasoning", False)
    g = SUTUIDGenerator(d)
    assert g.uid == "openai/chatgpt-4o:openai:openai_mod:y_reas:n_t:0.8_p:6_k:5"

    d.add("driver_code_version", "abcde")
    d.add("date", "20250723")
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723_dv:abcde_mod:y_reas:n_t:0.8_p:6_k:5"

    d.add("display_name", "my favorite SUT")
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723_dv:abcde_mod:y_reas:n_t:0.8_p:6_k:5_dn:my-favorite-sut"
