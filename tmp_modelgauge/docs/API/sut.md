::: modelgauge.sut
    options:
        show_root_heading: true
        show_root_toc_entry: false
        merge_init_into_class: true
        show_if_no_docstring: false
        show_bases: true
        heading_level: 1
        members:
            - SUT
            - PromptResponseSUT

::: modelgauge.sut
    options:
        show_root_toc_entry: false
        show_if_no_docstring: true
        show_bases: false
        members:
            - SUTResponse
            - SUTCompletion
            - TopTokens
            - TokenProbability
