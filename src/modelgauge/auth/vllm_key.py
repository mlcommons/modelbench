from modelgauge.secret_values import RequiredSecret, SecretDescription


class VllmApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="vllm",
            key="api_key",
            instructions="Contact MLCommons admin for access.",
        )
