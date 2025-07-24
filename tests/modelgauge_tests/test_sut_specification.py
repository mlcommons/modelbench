from modelgauge.sut_specification import SUTSpecification, SUTSpecificationChunk


def test_convenience_methods():
    s = SUTSpecification()
    assert s.requires("model")
    assert not s.requires("reasoning")

    assert s.knows("moderated")
    assert not s.knows("bogus")
