import boto3

from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.aws_bedrock_client import AmazonBedrockSut, AwsAccessKeyId, AwsSecretAccessKey


class AWSBedrockSUTFactory(DynamicSUTFactory):
    DRIVER_NAME = "aws"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None  # Lazy load.

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                service_name="bedrock",
                region_name="us-east-1",
                aws_access_key_id=self.injected_secrets()[0].value,
                aws_secret_access_key=self.injected_secrets()[1].value,
            )
        return self._client

    def _convert_model_id(self, model_id: str) -> SUTDefinition:
        """Convert AWS model IDs (maker.model[:version?]) to our standard format."""
        maker, model_name = model_id.split(".", maxsplit=1)
        model_name = model_name.replace(":", ".")
        return SUTDefinition({"maker": maker, "model": model_name, "driver": self.DRIVER_NAME})

    def _get_available_models(self, maker: str):
        response = self.client.list_foundation_models()
        models = {}
        for m in response["modelSummaries"]:
            if m.get("modelLifecycle", {}).get("status") != "ACTIVE":
                continue
            models[m["modelId"]] = self._convert_model_id(m["modelId"])
        return models

    def _get_model_id(self, sut_definition: SUTDefinition):
        models = self._get_available_models(sut_definition.to_dynamic_sut_metadata().maker)
        for model_id, model_definition in models.items():
            if str(model_definition.to_dynamic_sut_metadata()) == str(sut_definition.to_dynamic_sut_metadata()):
                return model_id
        supported_models = [model_def.to_dynamic_sut_metadata().external_model_name() for model_def in models.values()]
        raise ModelNotSupportedError(
            f"Model {sut_definition.external_model_name()} not found among AWS Bedrock models. AWS carries the following models from maker {sut_definition.get("maker")}: {supported_models} "
        )

    def get_secrets(self) -> list[InjectSecret]:
        return [InjectSecret(AwsAccessKeyId), InjectSecret(AwsSecretAccessKey)]

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        model_id = self._get_model_id(sut_definition)
        return AmazonBedrockSut(sut_definition.dynamic_uid, model_id, *self.injected_secrets())
