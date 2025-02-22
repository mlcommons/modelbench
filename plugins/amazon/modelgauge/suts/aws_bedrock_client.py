from abc import abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from modelgauge.general import APIException
from modelgauge.prompt import TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import REFUSAL_RESPONSE, PromptResponseSUT, RequestType, ResponseType, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

import boto3
import json

from modelgauge.suts.google_genai_client import GoogleAiApiKey


class AwsAccessKeyId(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="aws",
            key="access_key_id",
            instructions="See https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html",
        )


class AwsSecretAccessKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="aws",
            key="secret_access_key",
            instructions="See https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html",
        )


class BedrockRequest(BaseModel):
    class BedrockMessage(BaseModel):
        class InferenceConfig(BaseModel):
            maxTokens: Optional[int] = None
            temperature: Optional[float] = None
            topP: Optional[float] = None
            stopSequences: Optional[list[str]] = None

        class GuardrailConfig(BaseModel):
            guardrailIdentifier: str = None
            guardrailVersion: str = None
            trace: Optional[str] = None

        # as defined here: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
        role: str = "user"
        content: List[Dict[str, str]] = []
        system: Optional[List[Dict]] = None
        inferenceConfig: Optional[InferenceConfig] = None
        toolConfig: Optional[Dict] = None
        guardrailConfig: Optional[GuardrailConfig] = None
        additionalModelRequestFields: Optional[Any] = None
        promptVariables: Optional[Dict] = None
        additionalModelResponseFieldPaths: Optional[List[str]] = None
        requestMetadata: Optional[Dict] = None
        performanceConfig: Optional[Dict] = None

    modelId: str = None
    messages: List[BedrockMessage] = []


class BedrockResponse(BaseModel):
    class BedrockResponseMetadata(BaseModel):
        RequestId: str = None
        HTTPStatusCode: int = None
        HTTPHeaders: Dict[str, str] = None
        RetryAttempts: int = None

    class BedrockResponseOutput(BaseModel):
        class BedrockResponseMessage(BaseModel):
            class BedrockResponseContent(BaseModel):
                text: Optional[str] = None
                image: Optional[dict] = None
                document: Optional[dict] = None
                video: Optional[dict] = None
                toolUse: Optional[dict] = None
                toolResult: Optional[dict] = None
                guardContent: Optional[dict] = None

            role: str = None
            content: List[BedrockResponseContent] = []

        message: BedrockResponseMessage = None
        HTTPStatusCode: int = None

    class BedrockResponseUsage(BaseModel):
        inputTokens: int = None
        outputTokens: int = None
        totalTokens: int = None

    ResponseMetadata: BedrockResponseMetadata = None
    output: BedrockResponseOutput = None
    stopReason: str = None
    usage: BedrockResponseUsage = None
    metrics: dict = None
    additionalModelResponseFields: Optional[Any] = None
    trace: Optional[dict] = None
    performanceConfig: Optional[dict] = None


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class AmazonNovaSut(PromptResponseSUT[BedrockRequest, BedrockResponse]):

    def __init__(self, uid: str, model_id: str, access_key_id: AwsAccessKeyId, secret_access_key: AwsSecretAccessKey):
        super().__init__(uid)
        self.model_id = model_id
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=access_key_id.value,
            aws_secret_access_key=secret_access_key.value,
        )

    def translate_text_prompt(self, prompt: TextPrompt) -> BedrockRequest:
        inference_config = BedrockRequest.BedrockMessage.InferenceConfig(
            maxTokens=prompt.options.max_tokens,
            temperature=prompt.options.temperature,
            topP=prompt.options.top_p,
            stopSequences=prompt.options.stop_sequences,
        )

        return BedrockRequest(
            modelId=self.model_id,
            messages=[
                BedrockRequest.BedrockMessage(content=[{"text": prompt.text}]),
            ],
            inferenceConfig=inference_config,
        )

    def evaluate(self, request: BedrockRequest) -> BedrockResponse:
        response = self.client.converse(**request.model_dump(exclude_none=True))
        return BedrockResponse(**response)

    def translate_response(self, request: BedrockRequest, response: BedrockResponse) -> SUTResponse:
        completions = [SUTCompletion(text=x.text) for x in response.output.message.content]
        return SUTResponse(completions=completions)


BEDROCK_MODELS = ["micro", "lite", "pro"]

for model in BEDROCK_MODELS:
    SUTS.register(
        AmazonNovaSut,
        f"amazon-nova-1.0-{model}",
        f"amazon.nova-{model}-v1:0",
        InjectSecret(AwsAccessKeyId),
        InjectSecret(AwsSecretAccessKey),
    )
