import difflib
import re
from collections import defaultdict

from anthropic import Anthropic

from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import RawSecrets, InjectSecret
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.anthropic_api import AnthropicApiKey, AnthropicSUT


class AnthropicSUTFactory(DynamicSUTFactory):
    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(AnthropicApiKey)
        return [api_key]

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None  # Lazy load.

    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(api_key=self._secret().value)
        return self._client

    def _secret(self) -> AnthropicApiKey:
        return self.injected_secrets()[0]

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        model_names = [m.id for m in self.client().models.list()]
        uid = sut_definition.dynamic_uid
        requested_model = sut_definition.to_dynamic_sut_metadata().model
        if requested_model not in model_names:
            dateless_names = defaultdict(list)
            for n in model_names:
                key = re.sub(r"-\d{8}$", "", n)
                dateless_names[key].append(n)

            if requested_model not in dateless_names:
                raise ModelNotSupportedError(
                    f"{requested_model} not specific enough. Closest options are {difflib.get_close_matches(requested_model, model_names, cutoff=0.1)}"
                )

            if len(dateless_names[requested_model]) > 1:
                raise ModelNotSupportedError(
                    f"{requested_model} not specific enough. Available options are {dateless_names[requested_model]}"
                )

            new_name = dateless_names[requested_model][0]
            uid = uid.replace(requested_model, new_name)
            requested_model = new_name

        return AnthropicSUT(uid, requested_model, self._secret())
