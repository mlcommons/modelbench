from unittest.mock import patch

import pytest

from modelgauge.preflight import SUT_TYPE_DYNAMIC, SUT_TYPE_KNOWN, SUT_TYPE_UNKNOWN, classify_sut_uid, validate_sut_uid
from modelgauge.sut import SUT, SUTNotFoundException
from modelgauge.sut_registry import SUTS


@patch("modelgauge.preflight.make_dynamic_sut_for", autospec=True)  # prevents huggingface API lookups
def test_classify(patched):
    SUTS.register(SUT, "known", "something")
    assert classify_sut_uid("known") == SUT_TYPE_KNOWN
    assert classify_sut_uid("google:gemma:nebius:hfrelay") == SUT_TYPE_DYNAMIC
    assert classify_sut_uid("pleasedontregisterasutwiththisuid") == SUT_TYPE_UNKNOWN
    del SUTS._lookup["known"]


@patch(
    "modelgauge.preflight.make_dynamic_sut_for",
    autospec=True,
    return_value=(SUT, "google:gemma:nebius:hfrelay", "google:gemma:nebius:hfrelay"),
)
def test_validate_sut_uid(patched):
    assert validate_sut_uid(None) is None
    assert validate_sut_uid("") == ""
    with pytest.raises(SUTNotFoundException):
        _ = validate_sut_uid("bogus")

    SUTS.register(SUT, "known", "something")
    assert validate_sut_uid("known") == "known"
    del SUTS._lookup["known"]

    assert validate_sut_uid("google:gemma:nebius:hfrelay") == "google:gemma:nebius:hfrelay"
