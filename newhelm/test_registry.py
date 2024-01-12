from newhelm.base_test import BaseTest
from newhelm.instance_registery import InstanceRegistry

# The list of all Test instances with assigned names.
TESTS = InstanceRegistry[BaseTest]()
