from typing import Dict

from modelgauge.config import load_secrets_from_config

from modelgauge.secret_values import get_all_secrets, RawSecrets, RequiredSecret, SecretDescription


class FakeRequiredSecret(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(scope="some-scope", key="some-key", instructions="some-instructions")


def fake_all_secrets(value="some-value", use_real_secrets_for: list[str] | None = None) -> RawSecrets:
    secrets = get_all_secrets()
    raw_secrets: Dict[str, Dict[str, str]] = {}
    real_secrets = load_secrets_from_config()

    for secret in secrets:
        if secret.scope not in raw_secrets:
            raw_secrets[secret.scope] = {}
        if use_real_secrets_for and secret.scope in use_real_secrets_for:
            raw_secrets[secret.scope][secret.key] = real_secrets[secret.scope][secret.key]
        else:
            raw_secrets[secret.scope][secret.key] = value

    return raw_secrets
