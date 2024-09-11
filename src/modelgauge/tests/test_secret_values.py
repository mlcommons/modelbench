import pytest
from modelgauge.general import get_class
from modelgauge.secret_values import (
    InjectSecret,
    MissingSecretValues,
    OptionalSecret,
    RequiredSecret,
    SecretDescription,
    SerializedSecret,
    get_all_secrets,
)


class SomeRequiredSecret(RequiredSecret):
    @classmethod
    def description(cls):
        return SecretDescription(
            scope="some-scope", key="some-key", instructions="some-instructions"
        )


class SomeOptionalSecret(OptionalSecret):
    @classmethod
    def description(cls):
        return SecretDescription(
            scope="optional-scope",
            key="optional-key",
            instructions="optional-instructions",
        )


def test_descriptions():
    assert SomeRequiredSecret.description().scope == "some-scope"
    assert SomeOptionalSecret.description().scope == "optional-scope"


def test_make_required_present():
    secret = SomeRequiredSecret.make({"some-scope": {"some-key": "some-value"}})
    assert type(secret) == SomeRequiredSecret
    assert secret.value == "some-value"


def test_make_required_missing():
    with pytest.raises(MissingSecretValues) as err_info:
        secret = SomeRequiredSecret.make(
            {"some-scope": {"different-key": "some-value"}}
        )
    assert (
        str(err_info.value)
        == """\
Missing the following secrets:
scope='some-scope' key='some-key' instructions='some-instructions'
"""
    )


def test_make_optional_present():
    secret = SomeOptionalSecret.make({"optional-scope": {"optional-key": "some-value"}})
    assert type(secret) == SomeOptionalSecret
    assert secret.value == "some-value"


def test_make_optional_missing():
    secret = SomeOptionalSecret.make(
        {"optional-scope": {"different-key": "some-value"}}
    )
    assert secret.value is None


def test_missing_required_secrets_combine():
    secret1 = SecretDescription(scope="s1", key="k1", instructions="i1")
    secret2 = SecretDescription(scope="s2", key="k2", instructions="i2")
    e1 = MissingSecretValues([secret1])
    e2 = MissingSecretValues([secret2])

    combined = MissingSecretValues.combine([e1, e2])

    assert (
        str(combined)
        == """\
Missing the following secrets:
scope='s1' key='k1' instructions='i1'
scope='s2' key='k2' instructions='i2'
"""
    )


def test_get_all_secrets():
    descriptions = get_all_secrets()
    required_secret = SomeRequiredSecret.description()
    matching = [s for s in descriptions if s == required_secret]

    # This test can be impacted by other files, so just
    # check that at least one exists.
    assert len(matching) > 0, f"Found secrets: {descriptions}"


def test_serialize_secret():
    original = SomeRequiredSecret("some-value")
    serialized = SerializedSecret.serialize(original)
    assert serialized == SerializedSecret(
        module="tests.test_secret_values", class_name="SomeRequiredSecret"
    )
    returned = get_class(serialized.module, serialized.class_name)
    assert returned.description() == SecretDescription(
        scope="some-scope", key="some-key", instructions="some-instructions"
    )


def test_inject_required_present():
    injector = InjectSecret(SomeRequiredSecret)
    result = injector.inject({"some-scope": {"some-key": "some-value"}})
    assert result.value == "some-value"


def test_inject_required_missing():
    injector = InjectSecret(SomeRequiredSecret)
    with pytest.raises(MissingSecretValues):
        injector.inject({"some-scope": {"different-key": "some-value"}})
