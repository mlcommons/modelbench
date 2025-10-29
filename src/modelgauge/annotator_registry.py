from modelgauge.annotators.cheval.ids import SAFETY_ANNOTATOR_V1_1_UID, SECURITY_ANNOTATOR_V0_5_UID
from modelgauge.instance_factory import InstanceFactory
from modelgauge.annotator import Annotator

ANNOTATOR_MODULE_MAP = {
    "llama_guard_1": "llama_guard_annotator",
    "llama_guard_2": "llama_guard_annotator",
    "demo_annotator": "demo_annotator",
    "openai_compliance_annotator": "openai_compliance_annotator",
    "perspective_api": "perspective_api",
    SAFETY_ANNOTATOR_V1_1_UID: "cheval.registration",
    SECURITY_ANNOTATOR_V0_5_UID: "cheval.registration",
}

# The list of all Annotators instances with assigned UIDs.
ANNOTATORS = InstanceFactory[Annotator]()
