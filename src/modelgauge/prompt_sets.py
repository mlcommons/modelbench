from modelgauge.secret_values import RequiredSecret, SecretDescription
from pydantic import BaseModel

PROMPT_SET_DOWNLOAD_HOST = "ailuminate.mlcommons.org"


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
    locale: str
    filename: str

    @staticmethod
    def find(prompt_set_type: str, locale: str, prompt_sets: list) -> "PromptSet | None":
        for ps in prompt_sets:
            if ps.prompt_set_type == prompt_set_type and ps.locale == locale:
                return ps
        return None

    def url(self):
        return f"https://{PROMPT_SET_DOWNLOAD_HOST}/files/download/{self.filename}"


PROMPT_SETS = (
    PromptSet(prompt_set_type="practice", locale="en_us", filename="airr_official_1.0_practice_prompt_set_release.csv"),
    PromptSet(prompt_set_type="official", locale="en_us", filename="airr_official_1.0_heldback_prompt_set_release.csv"),
    PromptSet(
        prompt_set_type="practice", locale="fr_fr", filename="airr_official_1.0_practice_prompt_set_release_fr_fr.csv"
    ),
)
