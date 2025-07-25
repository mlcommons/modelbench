import re
from typing import Annotated, Optional

from pydantic import BaseModel, StringConstraints

SEPARATOR = ":"
RICH_UID_FIELD_SEPARATOR = ";"


class DynamicSUTSpecificationError(ValueError):
    """Use for a SUT UID or metadata that's not specified correctly, e.g.
    it's missing a required field. Only use this if the more specific
    exceptions defined in this module can't be used."""

    pass


class UnknownSUTDriverError(DynamicSUTSpecificationError):
    """Use when requesting a dynamic SUT that can't be created because we don't know
    how to talk to it"""

    pass


class MissingModelError(DynamicSUTSpecificationError):
    """Use for a SUT UID that has no specified model"""

    pass


def _is_date(s: str) -> bool:
    found = re.fullmatch(r"^\d{4}-?\d{2}-?\d{2}$", s)
    return found is not None


class DynamicSUTMetadata(BaseModel):
    """Elements that can be combined into a SUT UID.
    [maker/]model[:provider[:driver]][:date]
    E.g.
    mistralai/Mistral-Small-3.1-24B-Instruct-2503:nebius:hfrelay
    meta-llama/Llama-3.1-8B-Instruct:huggingface
    """

    model: Annotated[str, StringConstraints(strip_whitespace=True)]
    maker: Optional[str] = ""
    provider: Optional[str] = ""
    driver: str = ""
    date: Optional[str] = ""

    def is_proxied(self):
        blanks = (None, "")
        return self.driver not in blanks and self.provider not in blanks

    def external_model_name(self):
        if self.maker:
            return f"{self.maker}/{self.model}"
        return self.model

    @staticmethod
    def is_complete(sut_metadata: "DynamicSUTMetadata") -> bool:
        blanks = (None, "")
        return sut_metadata.driver not in blanks and sut_metadata.model not in blanks

    @staticmethod
    def parse_sut_uid(uid: str) -> "DynamicSUTMetadata":
        """Parses a SUT UID string to specify where a model runs and how we connect to it.

        Spec: [maker/]model:[provider:]driver[:date]
        Example: google/gemma-3-27b-it:nebius:hfrelay:20250507

        * driver == API client to connect to the ...
        * provider == service providing hardware resources to run the ...
        * model == name of the model, sometimes using the Huggingface naming scheme (e.g. meta/llama-xyz)

        You can specify a driver and no provider if the driver has a "default" provider, e.g. Together.

        It's possible to run a model on Together hardware and call it directly (using the Together client/driver),
        or indirectly via the Huggingface relay (using the Huggingface client/driver).

        Parsing rules:

        * split on colons
        * remove the date at the end if there is one
        * the first one is the maker/model
        * if there's only one chunk left, it's the native driver for a provider (e.g. Huggingface API, model running on Huggingface)
        * if there are two left, it's a provider (cohere, nebius) using the specified driver (API client) (e.g. huggingface relay)
        """

        def parse_model_name(m):
            if "/" in m:
                maker, model = m.split("/", 2)
            else:
                maker = ""
                model = m
            return maker, model

        metadata = DynamicSUTMetadata(model="", driver="")

        # rich SUT UIDs use ; to separate SUT option k:v from each other AND the dynamic SUT UID portion
        sut_uid_parts = uid.split(RICH_UID_FIELD_SEPARATOR)
        dynamic_uid = sut_uid_parts[0]
        chunks = dynamic_uid.split(SEPARATOR)
        if len(chunks) < 1 or len(chunks) > 4:
            raise DynamicSUTSpecificationError(f"{uid} is not a well-formed dynamic SUT UID.")

        # optional date suffix
        if _is_date(chunks[-1]):
            metadata.date = chunks[-1]
            del chunks[-1]

        # model is always present
        metadata.maker, metadata.model = parse_model_name(chunks[0])

        match len(chunks):
            # not proxied
            case 2:
                metadata.provider = ""
                metadata.driver = chunks[1]
            # proxied
            case 3:
                metadata.provider = chunks[1]
                metadata.driver = chunks[2]

        # try to support legacy SUT IDs
        if not metadata.driver:
            if "-hf" in dynamic_uid:
                metadata.driver = "huggingface"
            elif "-together" in dynamic_uid:
                metadata.driver = "together"

        if not DynamicSUTMetadata.is_complete(metadata):
            if not metadata.model:
                raise MissingModelError(f"SUT UID {dynamic_uid} is missing model")
            elif not metadata.driver:
                raise UnknownSUTDriverError(f"SUT UID {dynamic_uid} is missing driver")
            else:  # shouldn't happen
                raise DynamicSUTSpecificationError(f"Error parsing SUT UID {dynamic_uid}")

        return metadata

    @staticmethod
    def make_sut_uid(sut_metadata: "DynamicSUTMetadata") -> str:
        if not DynamicSUTMetadata.is_complete(sut_metadata):
            if not sut_metadata.model:
                raise MissingModelError(f"SUT specification is missing model")
            elif not sut_metadata.driver:
                raise UnknownSUTDriverError(f"SUT specification is missing model")
            else:  # shouldn't happen
                raise DynamicSUTSpecificationError(f"Bad SUT specification")

        # google/gemma-3-27b-it:nebius:hfrelay:20250507
        head = sut_metadata.external_model_name()

        chunks = [
            chunk
            for chunk in (
                head,
                sut_metadata.provider,
                sut_metadata.driver,
                sut_metadata.date,
            )
            if chunk
        ]
        return SEPARATOR.join(chunks)

    def __str__(self):
        return DynamicSUTMetadata.make_sut_uid(self)
