from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config
from modelgauge.dynamic_sut_finder import make_dynamic_sut_for
from modelgauge.secret_values import MissingSecretValues


from typing import List

from modelgauge.sut import SUTNotFoundException
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS


def listify(value):
    """Some functions accept a single UID or a list of them. This turns a single one into a list."""
    if isinstance(value, str):
        return [value]
    return value


def get_missing_secrets(secrets, registry, uids):
    missing_secrets: List[MissingSecretValues] = []
    for uid in uids:
        missing_secrets.extend(registry.get_missing_dependencies(uid, secrets=secrets))
    return missing_secrets


def check_secrets(secrets, sut_uids=None, test_uids=None, annotator_uids=None):
    """Checks if all secrets are present for the given UIDs. Raises an error and reports all missing secrets."""
    missing_secrets: List[MissingSecretValues] = []
    if sut_uids is not None:
        missing_secrets.extend(get_missing_secrets(secrets, SUTS, listify(sut_uids)))
    if test_uids is not None:
        missing_secrets.extend(get_missing_secrets(secrets, TESTS, test_uids))
        # Check secrets for the annotators in the test as well.
        for test_uid in test_uids:
            test_cls = TESTS._get_entry(test_uid).cls
            missing_secrets.extend(get_missing_secrets(secrets, ANNOTATORS, test_cls.get_annotators()))
    if annotator_uids is not None:
        missing_secrets.extend(get_missing_secrets(secrets, ANNOTATORS, annotator_uids))
    raise_if_missing_from_config(missing_secrets)
    return True


def validate_sut_uid(sut_uid):
    # A blank sut uid is OK for some invocations of modelgauge.
    # Commands where a non-blank sut uid is required must enforce that with click
    if not sut_uid:
        return sut_uid

    sut_type = classify_sut_uid(sut_uid)
    if sut_type == SUT_TYPE_KNOWN:
        pass
    elif sut_type == SUT_TYPE_DYNAMIC:
        dynamic_sut = make_dynamic_sut_for(sut_uid)  # a tuple that can be splatted for SUTS.register
        if dynamic_sut:
            SUTS.register(*dynamic_sut)
        else:
            raise SUTNotFoundException(f"{sut_uid} is not a valid dynamic SUT UID")
    else:
        raise SUTNotFoundException(f"{sut_uid} is not a valid SUT UID")

    return sut_uid


def make_sut(sut_uid: str):
    """Checks that user has all required secrets and returns instantiated SUT."""
    secrets = load_secrets_from_config()
    check_secrets(secrets, sut_uids=[sut_uid])
    sut = SUTS.make_instance(sut_uid, secrets=secrets)
    return sut
