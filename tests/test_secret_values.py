from newhelm.secret_values import (
    RequiredSecret,
    SecretDescription,
)


class MySecret(RequiredSecret):
    @classmethod
    def description(cls):
        return SecretDescription(
            scope="some-scope", key="some-key", instructions="some-instructions"
        )


# TODO
