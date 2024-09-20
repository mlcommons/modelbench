from modelgauge.secret_values import RequiredSecret, SecretDescription


class TogetherApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="together",
            key="api_key",
            instructions="See https://api.together.xyz/settings/api-keys",
        )
