from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

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


def demo_prompt_set_from_private_prompt_set(prompt_set: str) -> str | None:
    """In a test environment, we replace the practice or official prompt sets
    (which require auth) with matching demo prompt sets (which are public).
    This function returns the demo counterpart to a given practice or official prompt set."""
    found_locale = ""
    for prompt_set_type, prompt_sets in PROMPT_SETS.items():
        for locale, prompt_set_file_base_name in prompt_sets.items():
            print(f"target {prompt_set} looking at {prompt_set_file_base_name}")
            if prompt_set_file_base_name == prompt_set:
                found_locale = locale
                break

    if found_locale:
        return PROMPT_SETS["demo"].get(found_locale, None)
    return prompt_set


def prompt_set_from_url(source_url) -> str:
    """Given the source_url from a WebData object, returns the bare prompt set name
    without an extension or hostname"""
    try:
        chunks = urlparse(source_url)
        filename = Path(chunks.path).stem
        return filename
    except Exception as exc:
        return source_url


def demo_prompt_set_url(url: str) -> str:
    source_prompt_set = prompt_set_from_url(url)
    target_prompt_set = demo_prompt_set_from_private_prompt_set(source_prompt_set)
    target_url = url.replace(source_prompt_set, target_prompt_set)
    return target_url
