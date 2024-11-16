from modelgauge.secret_values import RequiredSecret, SecretDescription


class Lg3LoraVllmApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="lg3_lora_vllm",
            key="api_key",
            instructions="Contact MLCommons admin for access.",
        )


class Lg3LoraVllmEndpointUrl(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="lg3_lora_vllm",
            key="endpoint_url",
            instructions="Contact MLCommons admin for access.",
        )
