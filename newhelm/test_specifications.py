from collections.abc import Generator
from typing import Any, Dict, Mapping, Optional, Tuple
from pydantic import BaseModel, ValidationError
import tomli
import newhelm.tests.specifications
from importlib import resources


class Identity(BaseModel):
    uid: str
    version: Optional[str] = None
    display_name: str


class TestSpecification(BaseModel):
    source: str
    """Source is NOT in the toml file.
    
    For toml files, this is the path the file was loaded from.
    """
    identity: Identity

    # TODO The rest of the fields.
    __test__ = False


def load_module_toml_files(module) -> Generator[Tuple[str, dict[str, Any]], None, None]:
    """Find all toml files in the module and return their contents."""
    for path in resources.files(module).iterdir():
        if not path.is_file():
            continue
        if not path.name.endswith(".toml"):
            continue
        try:
            with path.open("rb") as f:
                # `load` expects binary, but `f` is the generic IO type.
                yield (str(path), tomli.load(f))  # type: ignore
        except Exception as e:
            raise Exception(f"While processing {path}.") from e


def load_test_specification_files(
    override_files: Optional[Generator[Tuple[str, dict[str, Any]], None, None]] = None
) -> Mapping[str, TestSpecification]:
    """Return TestSpecifications in newhelm.tests.specifications keyed by uid."""
    results: Dict[str, TestSpecification] = {}
    if override_files is not None:
        generator = override_files
    else:
        generator = load_module_toml_files(newhelm.tests.specifications)
    for source, raw in generator:
        if "source" in raw:
            raise AssertionError(
                f"File {source} should not include the "
                f"`source` variable as that changes during packaging."
            )
        raw["source"] = source
        try:
            parsed = TestSpecification.model_validate(raw, strict=True)
        except ValidationError as e:
            raise AssertionError(
                f"Could not parse {source} into TestSpecification."
            ) from e
        uid = parsed.identity.uid
        if uid in results:
            existing = results[uid].source
            raise AssertionError(
                f"Expected UID to be unique across files, "
                f"but {existing} and {source} both have uid={uid}."
            )
        results[uid] = parsed
    return results
