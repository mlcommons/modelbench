from modelgauge.uid_generator import SUTUIDGenerator


def test_uid():
    g = SUTUIDGenerator()
    g.add("model", "chatgpt-4o")
    g.add("maker", "openai")
    g.add("driver", "openai")
    g.add("provider", "openai")
    g.add("temperature", 0.8)
    g.add("top_p", 6)
    g.add("top_k", 5)
    g.add("moderated", True)
    g.add("reasoning", False)
    assert g.uid == "openai/chatgpt-4o:openai:openai_mod:y_reas:n_t:0.8_p:6_k:5"

    g.add("driver_code_version", "abcde")
    g.add("date", "20250723")
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723_dv:abcde_mod:y_reas:n_t:0.8_p:6_k:5"

    g.add("display_name", "my favorite SUT")
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723_dv:abcde_mod:y_reas:n_t:0.8_p:6_k:5_dn:my-favorite-sut"
