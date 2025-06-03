import re
from abc import ABC, abstractmethod

from modelgauge.secret_values import InjectSecret
from modelgauge.sut_metadata import SUTMetadata

SEPARATOR = ":"


class ModelNotSupportedError(Exception):
    """Use when requesting a dynamic SUT from a correct proxy (e.g. Huggingface)
    and a correct provider (e.g. nebius, cohere) that doesn't support that model."""

    pass


class ProviderNotFoundError(Exception):
    """Use when requesting a dynamic SUT from a correct proxy (e.g. Huggingface)
    with an unknown or inactive provider (e.g. nebius, cohere)."""

    pass


class UnknownProxyError(Exception):
    """Use when requesting a dynamic SUT that can't be created because the proxy
    isn't known, e.g. for now it's not hf"""

    pass


# non-exhaustive list of strings that identify service providers, drivers, and model makers
# used as hints to disambiguate a SUT UID if needed
KNOWN_PROVIDERS = {
    "hf",
    "cerebras",
    "falai",
    "fal-ai",
    "fireworks",
    "hfinference",
    "hf-inference",
    "hyperbolic",
    "together",
    "baseten",
    "azure",
    "cohere",
    "nebius",
    "mistralai",
    "vertexai",
    "novita",
    "sambanova",
    "replicate",
}
KNOWN_DRIVERS = {"hfrelay"}
KNOWN_VENDORS = {
    "openai",
    "google",
    "meta",
    "microsoft",
    "deepseek",
    "mistralai",
    "nvidia",
    "alibaba",
    "zhipu",
    "cohere",
    "ibm",
    "internlm",
    "ai2",
    "ai21labs",
    "01ai",
}


class DynamicSUTMaker(ABC):

    @staticmethod
    @abstractmethod
    def get_secrets() -> InjectSecret:
        pass

    @staticmethod
    @abstractmethod
    def find(name: str):
        pass

    @staticmethod
    def parse_sut_uid(uid: str) -> SUTMetadata:
        # google:gemma-3-27b-it:nebius:hfrelay:20250507
        # Parsing rules:
        # 1. split on colons and start at the right
        # 2. remove the date if there is one
        # 3. the next chunk before a colon is the driver
        # 4. the driver parses the rest

        metadata = SUTMetadata(model="blank")

        chunks = uid.split(SEPARATOR)
        if len(chunks) < 1 or len(chunks) > 5:
            raise ValueError(f"{uid} is not a well-formed SUT UID.")

        # optional date suffix
        if _is_date(chunks[-1]):
            metadata.date = chunks[-1]
            del chunks[-1]

        match len(chunks):
            # model only
            case 1:
                metadata.model = chunks[0]
            # everything
            case 4:
                metadata.vendor = chunks[0]
                metadata.model = chunks[1]
                metadata.provider = chunks[2]
                metadata.driver = chunks[3]
                return metadata
            # vendor + model
            # model + provider
            case 2:
                if chunks[1] in KNOWN_PROVIDERS:
                    metadata.model = chunks[0]
                    metadata.provider = chunks[1]
                elif chunks[0] in KNOWN_VENDORS:
                    metadata.vendor = chunks[0]
                    metadata.model = chunks[1]
                else:
                    raise ValueError(f"SUT UID {uid} is ambiguous.")
            # model + provider + driver
            # vendor + model + provider
            case 3:
                if chunks[2] in KNOWN_DRIVERS:
                    metadata.model = chunks[0]
                    metadata.provider = chunks[1]
                    metadata.driver = chunks[2]
                else:
                    metadata.vendor = chunks[0]
                    metadata.model = chunks[1]
                    metadata.provider = chunks[2]

        # TODO validate the field values
        return metadata

    @staticmethod
    def make_sut_uid(sut_metadata: SUTMetadata) -> str:
        # google:gemma-3-27b-it:nebius:hfrelay:20250507

        chunks = [
            chunk
            for chunk in (
                sut_metadata.vendor,
                sut_metadata.model,
                sut_metadata.provider,
                sut_metadata.driver,
                sut_metadata.date,
            )
            if chunk
        ]
        return SEPARATOR.join(chunks)


def _is_date(s: str) -> bool:
    found = re.fullmatch(r"^\d{4}-?\d{2}-?\d{2}$", s)
    return found is not None
