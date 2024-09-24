from modelgauge.secret_values import RequiredSecret, SecretDescription


class HuggingFaceInferenceToken(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="hugging_face",
            key="token",
            instructions="You can create tokens at https://huggingface.co/settings/tokens.",
        )
