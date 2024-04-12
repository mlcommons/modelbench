from abc import ABC, abstractmethod
from dataclasses import dataclass
from modelgauge.general import get_concrete_subclasses
from pydantic import BaseModel
from typing import Generic, List, Mapping, Optional, Sequence, Type, TypeVar


class SecretDescription(BaseModel):
    """How to look up a secret and how to get the value if you don't have it."""

    scope: str
    key: str
    instructions: str


RawSecrets = Mapping[str, Mapping[str, str]]
"""Convenience typing for how the secrets are read from a file."""


SecretType = TypeVar("SecretType", bound="Secret")


@dataclass(frozen=True)
class Secret(ABC):
    """Base class for all secrets."""

    @classmethod
    @abstractmethod
    def description(cls) -> SecretDescription:
        """Information about how to lookup/obtain the secret."""
        pass

    @classmethod
    @abstractmethod
    def make(cls: Type[SecretType], raw_secrets: RawSecrets) -> SecretType:
        """Read the secret value from `raw_secrets to make this class."""
        pass


def get_all_secrets() -> Sequence[SecretDescription]:
    """Return the descriptions of all possible secrets."""
    secrets = get_concrete_subclasses(Secret)  # type: ignore
    return [s.description() for s in secrets]


class SerializedSecret(BaseModel):
    """Hold a pointer to the secret class in a serializable form."""

    module: str
    class_name: str

    @staticmethod
    def serialize(secret: Secret) -> "SerializedSecret":
        """Create a SerializedSecret from a Secret"""
        return SerializedSecret(
            module=secret.__class__.__module__,
            class_name=secret.__class__.__qualname__,
        )


RequiredSecretType = TypeVar("RequiredSecretType", bound="RequiredSecret")


class RequiredSecret(Secret):
    """Base class for all required secrets."""

    def __init__(self, value: str):
        super().__init__()
        self._value = value

    @property
    def value(self) -> str:
        """Get the value of the secret."""
        return self._value

    @classmethod
    def make(
        cls: Type[RequiredSecretType], raw_secrets: RawSecrets
    ) -> RequiredSecretType:
        """Construct this class from the data in raw_secrets.

        Raises MissingSecretValues if desired secret is missing.
        """
        secret = cls.description()
        try:
            return cls(raw_secrets[secret.scope][secret.key])
        except KeyError:
            raise MissingSecretValues([secret])


class MissingSecretValues(LookupError):
    """Exception describing one or more missing required secrets."""

    def __init__(self, descriptions: Sequence[SecretDescription]):
        assert descriptions, "Must have at least 1 description to raise an error."
        self.descriptions = descriptions

    @staticmethod
    def combine(errors: Sequence["MissingSecretValues"]) -> "MissingSecretValues":
        """Combine multiple exceptions into one."""
        descriptions: List[SecretDescription] = []
        for error in errors:
            descriptions.extend(error.descriptions)
        return MissingSecretValues(descriptions)

    def __str__(self):
        message = "Missing the following secrets:\n"
        for d in self.descriptions:
            # TODO Make this nicer.
            message += str(d) + "\n"
        return message


OptionalSecretType = TypeVar("OptionalSecretType", bound="OptionalSecret")


class OptionalSecret(Secret):
    """Base class for all optional secrets."""

    def __init__(self, value: Optional[str]):
        super().__init__()
        self._value = value

    @property
    def value(self) -> Optional[str]:
        """Get the secret value, or None if it wasn't provided."""
        return self._value

    @classmethod
    def make(
        cls: Type[OptionalSecretType], raw_secrets: RawSecrets
    ) -> OptionalSecretType:
        """Construct this class from the data in raw_secrets.

        Sets value to None if desired secret is missing.
        """
        secret = cls.description()
        try:
            return cls(raw_secrets[secret.scope][secret.key])
        except KeyError:
            return cls(None)


_T = TypeVar("_T")


# TODO Consider moving these to dependency_injection.py
class Injector(ABC, Generic[_T]):
    """Base class for delayed injection of a value."""

    @abstractmethod
    def inject(self, raw_secrets: RawSecrets) -> _T:
        """Use `raw_secrets` to construct the object."""
        pass


class InjectSecret(Injector, Generic[SecretType]):
    def __init__(self, secret_class: Type[SecretType]):
        self.secret_class = secret_class

    def inject(self, raw_secrets: RawSecrets) -> SecretType:
        return self.secret_class.make(raw_secrets)

    def __repr__(self):
        return f"InjectSecret({self.secret_class.__name__})"
