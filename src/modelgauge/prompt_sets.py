from modelgauge.secret_values import RequiredSecret, SecretDescription


class ModellabFileDownloadToken(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="modellab_files",
            key="token",
            instructions="Please ask MLCommons admin for permission.",
        )


# file name format:
# {prefix}_{version}_{type}(_{locale})_prompt_set_release
PROMPT_SETS = {
    "practice": "airr_official_1.0_practice_prompt_set_release",
    "official": "airr_official_1.0_heldback_prompt_set_release",
    "practice_fr": "airr_official_1.0_practice_fr_fr_prompt_set_release",
}
TEST_PROMPT_SETS = {
    "fake-prompts": "fake-prompts",
}

PROMPT_SET_DOWNLOAD_HOST = "ailuminate.mlcommons.org"


def prompt_set_file_base_name(prompt_set: str) -> str:
    filename = PROMPT_SETS.get(prompt_set, TEST_PROMPT_SETS.get(prompt_set, ""))
    return filename


def validate_prompt_set(prompt_set: str) -> bool:
    filename = prompt_set_file_base_name(prompt_set)
    if not filename:
        raise ValueError(f"Invalid prompt set {prompt_set}. Must be one of {PROMPT_SETS.keys()}.")
    return True
