import pytest

from modelgauge.dynamic_sut_factory import DynamicSUTFactory
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret
from modelgauge_tests.test_secret_values import MissingSecretValues, SomeOptionalSecret, SomeRequiredSecret


class FakeDynamicFactory(DynamicSUTFactory):
    @staticmethod
    def get_secrets() -> list[InjectSecret]:
        return [InjectSecret(SomeRequiredSecret), InjectSecret(SomeOptionalSecret)]

    def make_sut(self, sut_metadata: DynamicSUTMetadata):
        pass


def test_injected_secrets():
    factory = FakeDynamicFactory(
        {"some-scope": {"some-key": "some-value"}, "optional-scope": {"optional-key": "optional-value"}}
    )
    secrets = factory.injected_secrets()
    assert len(secrets) == 2
    assert isinstance(secrets[0], SomeRequiredSecret)
    assert secrets[0].value == "some-value"
    assert isinstance(secrets[1], SomeOptionalSecret)
    assert secrets[1].value == "optional-value"


def test_injected_secrets_missing_optional():
    factory = FakeDynamicFactory({"some-scope": {"some-key": "some-value"}})
    secrets = factory.injected_secrets()
    assert len(secrets) == 2
    assert isinstance(secrets[0], SomeRequiredSecret)
    assert secrets[0].value == "some-value"
    assert isinstance(secrets[1], SomeOptionalSecret)
    assert secrets[1].value is None


def test_injected_secrets_missing_required():
    factory = FakeDynamicFactory({"optional-scope": {"optional-key": "optional-value"}})
    with pytest.raises(MissingSecretValues):
        factory.injected_secrets()
