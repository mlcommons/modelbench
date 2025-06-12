import re
from typing import Annotated, Optional

from pydantic import BaseModel, StringConstraints

SEPARATOR = ":"


def _is_date(s: str) -> bool:
    found = re.fullmatch(r"^\d{4}-?\d{2}-?\d{2}$", s)
    return found is not None


class DynamicSUTMetadata(BaseModel):
    """Elements that can be combined into a SUT UID.
    [maker:]model[:provider[:driver]][:date]
    [google:]gemma[:cohere[:hfrelay]][:20250701]
    """

    model: Annotated[str, StringConstraints(strip_whitespace=True)]
    maker: Optional[str] = ""
    provider: str = ""
    driver: Optional[str] = ""
    date: Optional[str] = ""

    def is_proxied(self):
        return self.driver is not None and self.driver != ""

    def external_model_name(self):
        if self.maker:
            return f"{self.maker}/{self.model}"
        return self.model

    @staticmethod
    def parse_sut_uid(uid: str) -> "DynamicSUTMetadata":
        # google/gemma-3-27b-it:nebius:hfrelay:20250507
        # Parsing rules:
        # 1. split on colons and start at the right
        # 2. remove the date if there is one
        # 3. the next chunk before a colon is the driver
        # 4. the driver parses the rest

        def parse_model_name(m):
            if "/" in m:
                maker, model = m.split("/", 2)
            else:
                maker = ""
                model = m
            return maker, model

        metadata = DynamicSUTMetadata(model="blank")

        chunks = uid.split(SEPARATOR)
        if len(chunks) < 1 or len(chunks) > 4:
            raise ValueError(f"{uid} is not a well-formed dynamic SUT UID.")

        # optional date suffix
        if _is_date(chunks[-1]):
            metadata.date = chunks[-1]
            del chunks[-1]

        # model is always present
        metadata.maker, metadata.model = parse_model_name(chunks[0])

        match len(chunks):
            # not proxied
            case 2:
                metadata.provider = chunks[1]
            # proxied
            case 3:
                metadata.provider = chunks[1]
                metadata.driver = chunks[2]

        # TODO validate the field values
        return metadata

    @staticmethod
    def make_sut_uid(sut_metadata: "DynamicSUTMetadata") -> str:
        # google:gemma-3-27b-it:nebius:hfrelay:20250507
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
