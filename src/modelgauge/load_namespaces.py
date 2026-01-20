"""
This namespace loader will discover and load all modules from modelgauge's suts,
annotators, runners, and tests directories.

To see this in action:

* uv sync
* uv run modelgauge list
"""

import importlib
import pkgutil
from types import ModuleType
from typing import Iterator, List

from tqdm import tqdm

import modelgauge
import modelgauge.annotators
import modelgauge.suts
import modelgauge.tests


def _iter_namespace(ns_pkg: ModuleType) -> Iterator[pkgutil.ModuleInfo]:
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def list_objects() -> List[str]:
    """Get a list of submodule names without attempting to import them."""
    module_names = []
    for ns in ["tests", "suts", "annotators"]:
        for _, name, _ in _iter_namespace(getattr(modelgauge, ns)):
            module_names.append(name)
    return module_names


modules_loaded = False


def load_namespaces(disable_progress_bar: bool = False) -> None:
    """Import all relevant modules."""
    global modules_loaded
    if not modules_loaded:
        modules = list_objects()
        for module_name in tqdm(
            modules,
            desc="Loading modules",
            disable=disable_progress_bar or len(modules) == 0,
        ):
            importlib.import_module(module_name)
        modules_loaded = True


def load_namespace(module_name: str) -> None:
    mod = importlib.import_module(f"modelgauge.{module_name}")
    if hasattr(mod, "__path__"):
        for _, name, _ in _iter_namespace(mod):
            importlib.import_module(name)
