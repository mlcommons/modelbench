from newhelm.general import get_unique_id


def test_unique_id():
    value = get_unique_id()
    assert isinstance(value, str)
    assert value != ""
