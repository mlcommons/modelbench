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
    "practice_fr_fr": "airr_official_1.0_practice_fr_fr_prompt_set_release",
    "official_fr_fr": "airr_official_1.0_heldback_fr_fr_prompt_set_release",
    "demo": "airr_official_1.0_demo_prompt_set_release",
}
PROMPT_SET_DOWNLOAD_HOST = "ailuminate.mlcommons.org"


def prompt_set_file_base_name(prompt_set: str, prompt_sets: dict = PROMPT_SETS) -> str:
    filename = prompt_sets.get(prompt_set, None)
    return filename


def validate_prompt_set(prompt_set: str, prompt_sets: dict = PROMPT_SETS) -> bool:
    filename = prompt_set_file_base_name(prompt_set, prompt_sets)
    if not filename:
        raise ValueError(f"Invalid prompt set {prompt_set}. Must be one of {prompt_sets.keys()}.")
    return True
