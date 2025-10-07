from modelgauge.annotator_set import BasicAnnotatorSet

try:
    from modelgauge.annotators.official import (
        SAFETY_ANNOTATOR_V1_0_UID,
        SAFETY_ANNOTATOR_V1_1_UID,
        SECURITY_ANNOTATOR_V0_5_UID,
    )
except:
    raise NotImplementedError("Private annotators are not available.")

PRIVATE_ANNOTATOR_SET = BasicAnnotatorSet(SAFETY_ANNOTATOR_V1_0_UID)
PRIVATE_ANNOTATOR_SET_V_1_1 = BasicAnnotatorSet(SAFETY_ANNOTATOR_V1_1_UID)
PRIVATE_SECURITY_ANNOTATOR_SET = BasicAnnotatorSet(SECURITY_ANNOTATOR_V0_5_UID)
