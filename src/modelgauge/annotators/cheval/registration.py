from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.cheval.annotator import ChevalAPIKey, ChevalAnnotator, ChevalEndpointUrl
from modelgauge.annotators.cheval.ids import SAFETY_ANNOTATOR_V1_1_UID, SECURITY_ANNOTATOR_V1_0_2_UID
from modelgauge.secret_values import InjectSecret

ANNOTATORS.register(
    ChevalAnnotator,
    SAFETY_ANNOTATOR_V1_1_UID,
    InjectSecret(ChevalAPIKey),
    InjectSecret(ChevalEndpointUrl),
)


ANNOTATORS.register(
    ChevalAnnotator,
    SECURITY_ANNOTATOR_V1_0_2_UID,
    InjectSecret(ChevalAPIKey),
    InjectSecret(ChevalEndpointUrl),
)
