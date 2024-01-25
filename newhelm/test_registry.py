from newhelm.base_test import BaseTest
from newhelm.instance_factory import InstanceFactory

# The list of all Test instances with assigned names.
TESTS = InstanceFactory[BaseTest]()
