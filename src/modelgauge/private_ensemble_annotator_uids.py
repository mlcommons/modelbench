# TODO: remove this file once modelbench-private is removed

try:
    from modelgauge.annotators.official import (
        SAFETY_ANNOTATOR_V1_0_UID,
        SAFETY_ANNOTATOR_V1_1_UID,
        SECURITY_ANNOTATOR_V0_5_UID,
    )
except:
    raise NotImplementedError("Private annotators are not available.")
