from modelgauge.instance_factory import InstanceFactory
from modelgauge.annotator import Annotator

# The list of all Annotators instances with assigned UIDs.
ANNOTATORS = InstanceFactory[Annotator]()
