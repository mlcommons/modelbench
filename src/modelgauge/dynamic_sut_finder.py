# Handles creating a dynamic SUT from its name. E.g. "google:gemma-3-27b-it:nebius:hfrelay:20250507"
# will create a gemma SUT hosted by nebius and proxied by huggingface.
# This is so the client code (e.g. modelbench benchmark run) doesn't need to know
# anything other than the SUT name/identifier passed by the user.


# Import the dynamic SUT builder modules here.
# Eventually, if we have more kinds, we should discover and auto-load those
# from the plugins directory like the load_plugins function auto-loads plugins
import modelgauge.suts.huggingface_sut_factory as hsf

from modelgauge.dynamic_sut_factory import UnknownSUTProviderError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.suts.together_sut_factory import TogetherSUTFactory

# Maps a string to the module and factory function in that module
# that can be used to create a dynamic sut
DYNAMIC_SUT_FACTORIES = {
    "proxied": {"hfrelay": hsf.make_sut},
    "direct": {"together": TogetherSUTFactory.make_sut, "huggingface": hsf.make_sut, "hf": hsf.make_sut},
}


def make_dynamic_sut_for(sut_name: str, *args, **kwargs):
    sut_metadata: DynamicSUTMetadata = DynamicSUTMetadata.parse_sut_uid(sut_name)

    if sut_metadata.driver:
        factory = DYNAMIC_SUT_FACTORIES["proxied"].get(sut_metadata.driver, None)
    else:
        factory = DYNAMIC_SUT_FACTORIES["direct"].get(sut_metadata.provider, None)  # type: ignore
    if not factory:
        raise UnknownSUTProviderError(f'Don\'t know how to make dynamic sut "{sut_name}"')
    return factory(sut_name, *args, **kwargs)
