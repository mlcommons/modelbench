"""
This is an example of a very simple namespace plugin loader that will discover and load all plugins from newhelm/plugins
and will declare and run all of those discovered plugins.

There's an example of a plugin at newhelm/mmlu and a commented out installation of that plugin in the newhelm/pyproject.toml
file. If you install that and run the below, you'll see that it picked up the new plugin and loaded it.
"""
import importlib
import pkgutil
from types import ModuleType
from typing import Iterator

import newhelm.plugins
import newhelm.plugins.benchmarks
import newhelm.plugins.suts
import newhelm.plugins.tests
import newhelm.plugins.runners


def iter_namespace(ns_pkg: ModuleType) -> Iterator[pkgutil.ModuleInfo]:
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def load_plugins() -> None:
    for ns in ["tests", "suts", "benchmarks", "runners"]:
        for _, name, _ in iter_namespace(getattr(newhelm.plugins, ns)):
            print(f"Importing: {name}")
            importlib.import_module(name)


if __name__ == "__main__":
    load_plugins()
    for ns in ["tests", "suts", "benchmarks", "runners"]:
        print(f"These are the {ns} I know about:")
        for plugin in list(pkgutil.iter_modules(getattr(newhelm.plugins, ns).__path__)):
            print(plugin.name)

        print(f"Here's me running the {ns} I know about:")
        for plugin in list(pkgutil.iter_modules(getattr(newhelm.plugins, ns).__path__)):
            getattr(newhelm.plugins.tests, plugin.name).run()

        print("\n")
