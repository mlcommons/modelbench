from newhelm.instance_registery import InstanceRegistry
from newhelm.sut import SUT

# The list of all SUT instances with assigned names.
SUTS = InstanceRegistry[SUT]()
