"""
This namespace plugin loader will discover and load all plugins from modelgauge's plugin directories.

To see this in action:

* poetry install
* poetry run modelgauge list
* poetry install --extras demo
* poetry run modelgauge list

The demo plugin modules will only print on the second run.
"""

import importlib
import pkgutil
from types import ModuleType
from typing import Iterator, List

from tqdm import tqdm

import modelgauge
import modelgauge.annotators
import modelgauge.runners
import modelgauge.suts
import modelgauge.tests


def _iter_namespace(ns_pkg: ModuleType) -> Iterator[pkgutil.ModuleInfo]:
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def list_plugins() -> List[str]:
    """Get a list of plugin module names without attempting to import them."""
    module_names = []
    for ns in ["tests", "suts", "runners", "annotators"]:
        for _, name, _ in _iter_namespace(getattr(modelgauge, ns)):
            module_names.append(name)
    return module_names


plugins_loaded = False


def load_plugins(disable_progress_bar: bool = False) -> None:
    """Import all plugin modules."""
    global plugins_loaded
    if not plugins_loaded:
        plugins = list_plugins()
        for module_name in tqdm(
            plugins,
            desc="Loading plugins",
            disable=disable_progress_bar or len(plugins) == 0,
        ):
            importlib.import_module(module_name)
        plugins_loaded = True
