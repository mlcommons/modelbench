from newhelm.instance_factory import InstanceFactory
from newhelm.sut import SUT

# The list of all SUT instances with assigned names.
SUTS = InstanceFactory[SUT]()
