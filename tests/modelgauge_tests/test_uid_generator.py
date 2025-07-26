from modelgauge.sut_definition import SUTUIDGenerator, SUTDefinition


def test_uid():
    d = SUTDefinition()
    d.add("model", "chatgpt-4o")
    d.add("maker", "openai")
    d.add("driver", "openai")
    d.add("provider", "openai")
    d.add("date", None)  # workaround for the test in test_to_dynamic_sut_metadata setting the date (?!)
    d.add("max_tokens", 500)
    d.add("temp", 0.8)
    d.add("top_p", 6)
    d.add("top_k", 5)
    d.add("top_logprobs", 1)
    d.add("moderated", True)
    d.add("reasoning", False)
    g = SUTUIDGenerator(d)
    assert g.uid == "openai/chatgpt-4o:openai:openai;mod:y;reas:n;mt:500;t:0.8;p:6;k:5;l:1"

    d.add("driver_code_version", "abcde")
    d.add("date", "20250723")
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723;dv:abcde;mod:y;reas:n;mt:500;t:0.8;p:6;k:5;l:1"

    d.add("display_name", "my favorite SUT")
    assert (
        g.uid
        == "openai/chatgpt-4o:openai:openai:20250723;dv:abcde;mod:y;reas:n;mt:500;t:0.8;p:6;k:5;l:1;dn:my_favorite_sut"
    )
