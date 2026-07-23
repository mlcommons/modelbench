from modelgauge.annotators.cheval.annotator import (
    ChevalAnnotator,
    ChevalAPIKey,
    ChevalEndpointUrl,
)
from modelgauge.annotators.cheval.ids import SAFETY_ANNOTATOR_V1_1_UID
from modelgauge.annotators.composed_annotator import (
    AnnotatorArbiter,
    Safety,
    SafetyDAGAnnotator,
)
from modelgauge.annotators.composer.dag import Composer


class SafetyDAGChevalAnnotator(SafetyDAGAnnotator):

    def __init__(
        self,
        uid: str,
        api_key: ChevalAPIKey,
        endpoint_url: ChevalEndpointUrl,
    ) -> None:
        cheval_annotator = ChevalAnnotator(
            uid=SAFETY_ANNOTATOR_V1_1_UID,
            api_key=api_key,
            endpoint_url=endpoint_url,
        )
        dag = Composer(name=uid, verdict_type=Safety).add_node(
            AnnotatorArbiter(name="cheval_safety", annotator=cheval_annotator)
        )
        super().__init__(uid, dag)
