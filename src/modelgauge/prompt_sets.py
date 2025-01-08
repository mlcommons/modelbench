from enum import Enum

from modelgauge.secret_values import RequiredSecret, SecretDescription
from pydantic import BaseModel

PROMPT_SET_DOWNLOAD_HOST = "ailuminate.mlcommons.org"
PROMPT_SET_DEFAULT_VERSION = "1.0"
PROMPT_SET_TYPES = "practice"


class PromptSetType(str, Enum):
    PRACTICE = "practice"
    OFFICIAL = "official"

    def __str__(self):
        return str(self.value)


class ModellabFileDownloadToken(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="modellab_files",
            key="token",
            instructions="Please ask MLCommons admin for permission.",
        )


class PromptSet(BaseModel):
    prompt_set_type: str
    locale: str  # may be none, for files containing multiple locales
    version: str
    filename: str

    @staticmethod
    def find(
        prompt_set: "PromptSet",
        prompt_sets: list = [],
    ) -> "PromptSet | None":
        for ps in prompt_sets:
            if (
                ps.prompt_set_type == prompt_set.prompt_set_type
                and ps.locale == prompt_set.locale
                and ps.version == prompt_set.version
            ):
                return ps
        return None

    @staticmethod
    def find_by(
        prompt_set_type: str,
        locale: str | None,
        version: str = PROMPT_SET_DEFAULT_VERSION,
        prompt_sets: list = [],
    ) -> "PromptSet | None":
        for ps in prompt_sets:
            if ps.prompt_set_type == prompt_set_type and ps.locale == locale and ps.version == version:
                return ps
        return None

    def url(self):
        return f"https://{PROMPT_SET_DOWNLOAD_HOST}/files/download/{self.filename}"


PROMPT_SETS = [
    PromptSet(
        prompt_set_type=PromptSetType.PRACTICE,
        locale="en_us",
        version="1.0",
        filename="airr_official_1.0_practice_prompt_set_release.csv",
    ),
    PromptSet(
        prompt_set_type=PromptSetType.OFFICIAL,
        locale="en_us",
        version="1.0",
        filename="airr_official_1.0_heldback_prompt_set_release.csv",
    ),
    PromptSet(
        prompt_set_type=PromptSetType.PRACTICE,
        locale="fr_fr",
        version="1.0",
        filename="airr_official_1.0_practice_prompt_set_release_fr_fr.csv",
    ),
]
