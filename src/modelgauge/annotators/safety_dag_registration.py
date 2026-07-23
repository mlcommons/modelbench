from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.cheval.annotator import ChevalAPIKey, ChevalEndpointUrl
from modelgauge.annotators.cheval.ids import SAFETY_ANNOTATOR_V1_1_1_UID
from modelgauge.annotators.safety_dag import SafetyDAGChevalAnnotator
from modelgauge.secret_values import InjectSecret

ANNOTATORS.register(
    SafetyDAGChevalAnnotator,
    SAFETY_ANNOTATOR_V1_1_1_UID,
    InjectSecret(ChevalAPIKey),
    InjectSecret(ChevalEndpointUrl),
)
