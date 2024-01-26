from abc import ABC, abstractmethod
from typing import Any, Dict

from newhelm.general import subset_dict


class RequiresCredentials(ABC):
    """Mixin for classes that require secret information."""

    @abstractmethod
    def get_credential_instructions(self) -> Dict[str, str]:
        """For each secret this object needs, report how to obtain it."""
        pass

    @abstractmethod
    def load_credentials(self, secrets_dict: Dict[str, str]) -> None:
        """Read from the secrets_dict to set variables on `self`.

        Only secrets listed in get_credential_instructions will be passed.
        """


def optionally_load_credentials(obj: Any, all_secrets_dict: Dict[str, str]) -> None:
    """Will handle loading credentials for obj, if it needs credentials."""
    if not isinstance(obj, RequiresCredentials):
        return
    instructions_dict = obj.get_credential_instructions()
    secrets_dict = subset_dict(all_secrets_dict, instructions_dict.keys())
    try:
        obj.load_credentials(secrets_dict)
    except KeyError as e:
        instructions = "\n".join(
            f"{key}: {text}" for key, text in instructions_dict.items()
        )
        raise AssertionError(
            "Ensure you have set the required keys:\n" + instructions
        ) from e
