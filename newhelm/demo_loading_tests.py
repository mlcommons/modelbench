from newhelm.base_test import BaseTest
from newhelm.load_plugins import load_plugins
from newhelm.general import get_concrete_subclasses

if __name__ == "__main__":
    load_plugins()
    for test_cls in get_concrete_subclasses(BaseTest):  # type: ignore[type-abstract]
        print("Fully qualified name of the test:", test_cls)
        print("Running that test:")
        test_cls().run()
