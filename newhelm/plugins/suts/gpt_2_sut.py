from newhelm.huggingface_client import HuggingFaceSUT
from newhelm.sut_registry import SUTS

SUTS.register("gpt2", HuggingFaceSUT, "gpt2")
