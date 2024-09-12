from modelgauge.base_test import BaseTest
from modelgauge.instance_factory import InstanceFactory

# The list of all Test instances with assigned UIDs.
TESTS = InstanceFactory[BaseTest]()
