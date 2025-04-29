import json
import pathlib
import sys
from pydantic import BaseModel

from modelbench.cache import DiskCache, MBCache

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.load_plugins import load_plugins
from modelgauge.config import load_secrets_from_config
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS

# Setup
load_plugins()
secrets = load_secrets_from_config()

hazards = ["vcr", "ncr", "src", "cse", "dfm", "spc", "prv", "ipv", "iwp", "hte", "ssh", "sxc"]
a_uid = "mistral-7b-ruby"
a = ANNOTATORS.make_instance(a_uid, secrets=secrets)
sut_uid = "nvidia-llama-3-1-nemotron-nano-8b-v1"
s = SUTS.make_instance(sut_uid, secrets=secrets)

# Make caches
data_dir = pathlib.Path("./run")
sut_cache = DiskCache(data_dir / "sut_cache")
annotator_cache = DiskCache(data_dir / "copy_annotator_cache")


def search_cache(key):
    if key in annotator_cache:
        print("Found item in annotator cache")
    else:
        print("Could not find item... caching")
        # annotator_cache[key] = "annotation"


def make_sut_cache_key(sut_request, sut_uid):
    request = sut_request.model_dump(exclude_none=True)
    json_key = {"sut": sut_uid, "sut_request": request}
    return json.dumps(json_key)

def make_annotator_cache_key(annotator_request, annotator_uid):
    # Add annotator UID to key to avoid collisions.
    json_key = {"annotator": annotator_uid}
    if isinstance(annotator_request, BaseModel):
        json_key["annotator_request"] = annotator_request.model_dump(exclude_none=True)
    elif isinstance(annotator_request, str):
        json_key["annotator_request"] = annotator_request
    else:
        raise ValueError(f"Don't know how to make a key out of {annotator_request.__class__}: {annotator_request}")
    return json.dumps(json_key)


num_found = 0
num_missing = 0
for h in hazards:
    t = TESTS.make_instance(f"safe-{h}-en_us-practice-1.0-ensemble", secrets=secrets)
    dep_helper = FromSourceDependencyHelper(data_dir / "tests", t.get_dependencies(), required_versions={})
    all_items = t.make_test_items(dep_helper)

    for item in all_items:
        # Get SUT Response
        raw_request = s.translate_text_prompt(item.prompt, t.sut_options())
        sut_cache_key = make_sut_cache_key(raw_request, sut_uid)

        assert sut_cache_key in sut_cache
        raw_response = sut_cache[sut_cache_key]
        response = s.translate_response(raw_request, raw_response)

        # Get annotator response
        annotator_request = a.translate_request(item, response)
        a_cache_key = make_annotator_cache_key(annotator_request, a_uid)
        if a_cache_key in annotator_cache:
            num_found += 1
        else:
            num_missing += 1
        # search_cache(a_cache_key)

print(f"Found {num_found} items in the annotator cache")
print(f"Missing {num_missing} items in the annotator cache")
print("Annotator cache size: ", len(annotator_cache))
