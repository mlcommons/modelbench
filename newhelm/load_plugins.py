"""
This namespace plugin loader will discover and load all plugins from newhelm's plugin directories.

To see this in action:

* poetry install
* poetry run python newhelm/main.py list
* poetry install --extras demo
* poetry run python newhelm/main.py list

The demo plugin modules will only print on the second run.
"""
import importlib
import pkgutil
from types import ModuleType
from typing import Iterator, List

import newhelm
import newhelm.annotators
import newhelm.suts
import newhelm.tests
import newhelm.runners


def _iter_namespace(ns_pkg: ModuleType) -> Iterator[pkgutil.ModuleInfo]:
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def list_plugins() -> List[str]:
    module_names = []
    for ns in ["tests", "suts", "runners", "annotators"]:
        for _, name, _ in _iter_namespace(getattr(newhelm, ns)):
            module_names.append(name)
    return module_names


def load_plugins() -> None:
    for module_name in list_plugins():
        importlib.import_module(module_name)
