import re
from typing import Annotated, Optional

from modelgauge.dynamic_sut_maker import KNOWN_DRIVERS, KNOWN_PROVIDERS, KNOWN_VENDORS

from pydantic import BaseModel, StringConstraints

SEPARATOR = ":"


class SUTMetadata(BaseModel):
    """Elements that can be combined into a SUT UID.
    [vendor:]model[:provider[:driver]][:date]
    [google:]gemma[:cohere[:hfrelay]][:20250701]
    """

    model: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9-_.]+$")]
    vendor: Optional[str] = ""
    provider: str = ""
    driver: Optional[str] = ""
    date: Optional[str] = ""

    def is_proxied(self):
        return self.driver is not None and self.driver != ""

    def external_model_name(self):
        if self.vendor:
            return f"{self.vendor}/{self.model}"
        return self.model

    @staticmethod
    def parse_sut_uid(uid: str) -> "SUTMetadata":
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
    def make_sut_uid(sut_metadata: "SUTMetadata") -> str:
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
