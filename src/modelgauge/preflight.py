from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config
from modelgauge.secret_values import MissingSecretValues


from typing import List

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


def make_sut(sut_uid: str):
    """Checks that user has all required secrets and returns instantiated SUT."""
    secrets = load_secrets_from_config()
    check_secrets(secrets, sut_uids=[sut_uid])
    sut = SUTS.make_instance(sut_uid, secrets=secrets)
    return sut
