from modelgauge.secret_values import OptionalSecret, RequiredSecret, SecretDescription


class OpenAICompatibleApiKey(RequiredSecret):
    provider: str = "unspecified"

    @classmethod
    def for_provider(cls, provider):
        cls.provider = provider
        return cls

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=cls.provider,
            key="api_key",
            instructions="See https://platform.openai.com/api-keys",
        )


class OpenAICompatibleOrganization(OptionalSecret):
    provider: str = "unspecified"

    @classmethod
    def for_provider(cls, provider):
        cls.provider = provider
        return cls

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=cls.provider,
            key="organization",
            instructions="See https://platform.openai.com/account/organization",
        )


class OpenAICompatibleBaseURL(OptionalSecret):
    """Technically not a secret, but using the existing secret machinery for expediency"""

    provider: str = "unspecified"

    @classmethod
    def for_provider(cls, provider):
        cls.provider = provider
        return cls

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=cls.provider,
            key="base_url",  # for OpenAI-compatible models running on a different provider
            instructions="https://github.com/openai/openai-python/blob/main/src/openai/_client.py",
        )


class OpenAIApiKey(OpenAICompatibleApiKey):
    provider = "openai"


class OpenAIOrganization(OpenAICompatibleOrganization):
    provider = "openai"


class OpenAIBaseURL(OpenAICompatibleBaseURL):
    provider = "openai"
