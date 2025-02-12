from typing import Any, Optional

from modelgauge.locales import EN_US
from modelgauge.secret_values import OptionalSecret, SecretDescription


class ModellabFileDownloadToken(OptionalSecret):
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
    "practice": {
        "en_us": "airr_official_1.0_practice_prompt_set_release",
        "fr_fr": "airr_official_1.0_practice_fr_fr_prompt_set_release",
    },
    "official": {
        "en_us": "airr_official_1.0_heldback_prompt_set_release",
        "fr_fr": "airr_official_1.0_heldback_fr_fr_prompt_set_release",
    },
    "demo": {
        "en_us": "airr_official_1.0_demo_prompt_set_release",
        "fr_fr": "airr_official_1.0_demo_fr_fr_prompt_set_release",
    },
}
PROMPT_SET_DOWNLOAD_HOST = "ailuminate.mlcommons.org"


def _flatten(prompt_sets: dict = PROMPT_SETS) -> str:
    options = set()
    for set_type, sets in prompt_sets.items():
        for locale in sets.keys():
            options.add(f"{set_type} + {locale}")
    sorted(options, reverse=True)
    return ", ".join(sorted(options, reverse=True))


def prompt_set_file_base_name(prompt_set: str, locale: str = EN_US, prompt_sets: dict = PROMPT_SETS) -> str:
    filename = None
    try:
        filename = prompt_sets[prompt_set][locale]
    except KeyError as exc:
        raise ValueError from exc
    return filename


def validate_prompt_set(prompt_set: str, locale: str = EN_US, prompt_sets: dict = PROMPT_SETS) -> bool:
    filename = prompt_set_file_base_name(prompt_set, locale, prompt_sets)
    if not filename:
        raise ValueError(
            f"Invalid prompt set {prompt_set} {locale}. Must be one of {prompt_sets.keys()} and {_flatten(prompt_sets)}."
        )
    return True


def prompt_set_to_filename(prompt_set: str) -> str:
    """The official, secret prompt set files are named .+_heldback_*, not _official_"""
    return prompt_set.replace("official", "heldback")


def validate_token_requirement(prompt_set: str, token=None) -> bool:
    """This does not validate the token itself, only its presence."""
    if prompt_set == "demo":
        return True
    if token:
        return True
    raise ValueError(f"Prompt set {prompt_set} requires a token from MLCommons.")
