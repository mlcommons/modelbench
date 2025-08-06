from modelgauge.instance_factory import InstanceFactory
from modelgauge.annotator import Annotator

ANNOTATOR_MODULE_MAP = {
    "llama_guard_1": "llama_guard_annotator",
    "llama_guard_2": "llama_guard_annotator",
    "demo_annotator": "demo_annotator",
    "openai_compliance_annotator": "openai_compliance_annotator",
    "perspective_api": "perspective_api",
}

# The list of all Annotators instances with assigned UIDs.
ANNOTATORS = InstanceFactory[Annotator]()
