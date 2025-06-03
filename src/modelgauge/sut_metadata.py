from typing import Annotated, Optional

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
