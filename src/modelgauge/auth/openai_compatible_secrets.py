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


class OpenAIOrganization(OptionalSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="openai",
            key="organization",
            instructions="See https://platform.openai.com/account/organization",
        )


class OpenAIApiKey(OpenAICompatibleApiKey):
    provider = "openai"
