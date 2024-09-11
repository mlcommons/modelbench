from modelgauge.instance_factory import InstanceFactory
from modelgauge.sut import SUT

# The list of all SUT instances with assigned UIDs.
SUTS = InstanceFactory[SUT]()
