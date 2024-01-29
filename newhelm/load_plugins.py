"""
This is an example of a very simple namespace plugin loader that will discover and load all plugins from newhelm/plugins
and will declare and run all of those discovered plugins.

To see this in action:

* poetry install
* poetry run python newhelm/load_plugins.py
* poetry install --extras demo
* poetry run python newhelm/load_plugins.py

The demo plugin modules will only print on the second run.
"""
import importlib
import pkgutil
from types import ModuleType
from typing import Iterator

import newhelm.plugins
import newhelm.plugins.annotators
import newhelm.plugins.benchmarks
import newhelm.plugins.suts
import newhelm.plugins.tests
import newhelm.plugins.runners


def iter_namespace(ns_pkg: ModuleType) -> Iterator[pkgutil.ModuleInfo]:
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def load_plugins() -> None:
    for ns in ["tests", "suts", "benchmarks", "runners", "annotators"]:
        for _, name, _ in iter_namespace(getattr(newhelm.plugins, ns)):
            print(f"Importing: {name}")
            importlib.import_module(name)


if __name__ == "__main__":
    load_plugins()
    for ns in ["tests", "suts", "benchmarks", "runners", "annotators"]:
        print(f"These are the {ns} modules I know about:")
        for plugin in list(pkgutil.iter_modules(getattr(newhelm.plugins, ns).__path__)):
            print(plugin.name)
        print("\n")
