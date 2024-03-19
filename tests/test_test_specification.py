from typing import Any, Tuple

import pytest
from newhelm.test_specifications import (
    Identity,
    TestSpecification,
    load_test_specification_files,
)


def _make_spec(source, uid) -> TestSpecification:
    return TestSpecification(
        source=source, identity=Identity(uid=uid, display_name="some display")
    )


def _mock_file(spec: TestSpecification) -> Tuple[str, dict[str, Any]]:
    raw = spec.model_dump()
    source = raw.pop("source")
    return (source, raw)


def test_load_basic():
    expected = _make_spec("some/path", "uid1")

    results = load_test_specification_files([_mock_file(expected)])
    assert results == {"uid1": expected}


def test_load_bad_value():
    expected = _make_spec("some/path", "uid1")
    mocked = _mock_file(expected)
    # Make identity wrong
    mocked[1]["identity"] = "some-value"

    with pytest.raises(AssertionError) as err_info:
        load_test_specification_files([mocked])
    assert str(err_info.value) == "Could not parse some/path into TestSpecification."
    # Ensure it forwards the validation issue.
    assert "valid dictionary or instance of Identity" in str(err_info.value.__cause__)


def test_load_should_not_include_source():
    expected = _make_spec("some/path", "uid1")
    mocked = _mock_file(expected)
    # Make identity wrong
    mocked[1]["source"] = "wrong/path"

    with pytest.raises(AssertionError) as err_info:
        load_test_specification_files([mocked])
    assert (
        str(err_info.value) == "File some/path should not include the "
        "`source` variable as that changes during packaging."
    )


def test_load_multiple():
    expected1 = _make_spec("p1", "uid1")
    expected2 = _make_spec("p2", "uid2")
    expected3 = _make_spec("p3", "uid3")

    results = load_test_specification_files(
        [
            _mock_file(expected1),
            _mock_file(expected2),
            _mock_file(expected3),
        ]
    )
    assert results == {
        "uid1": expected1,
        "uid2": expected2,
        "uid3": expected3,
    }


def test_load_repeated_uid():
    expected1 = _make_spec("p1", "uid1")
    expected2 = _make_spec("p2", "uid2")
    expected1_again = _make_spec("p3", "uid1")

    with pytest.raises(AssertionError) as err_info:
        load_test_specification_files(
            [
                _mock_file(expected1),
                _mock_file(expected2),
                _mock_file(expected1_again),
            ]
        )
    assert str(err_info.value) == (
        "Expected UID to be unique across files, " "but p1 and p3 both have uid=uid1."
    )


def test_load_module_no_error():
    # We don't know what files might exist, so just verify it runs.
    load_test_specification_files()
