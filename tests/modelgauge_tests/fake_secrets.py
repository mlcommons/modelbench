from typing import Dict

from modelgauge.secret_values import get_all_secrets, RawSecrets, RequiredSecret, SecretDescription


class FakeRequiredSecret(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(scope="some-scope", key="some-key", instructions="some-instructions")


def fake_all_secrets(value="some-value") -> RawSecrets:
    secrets = get_all_secrets()
    raw_secrets: Dict[str, Dict[str, str]] = {}

    for secret in secrets:
        if secret.scope not in raw_secrets:
            raw_secrets[secret.scope] = {}
        raw_secrets[secret.scope][secret.key] = value

    return raw_secrets
