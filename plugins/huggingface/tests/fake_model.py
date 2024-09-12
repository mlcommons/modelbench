from unittest.mock import MagicMock
from typing import List, Union

import torch
from transformers import BatchEncoding  # type: ignore

from modelgauge.suts.huggingface_client import (
    HuggingFaceSUT,
    HuggingFaceToken,
    WrappedPreTrainedTokenizer,
)


def make_client():
    return HuggingFaceSUT(
        uid="test-sut",
        pretrained_model_name_or_path="some-model",
        token=HuggingFaceToken("some-value"),
    )


def make_mocked_client(vocab_map, **t_kwargs):
    mock_model = MagicMock()
    client = make_client()
    client.wrapped_tokenizer = WrappedPreTrainedTokenizer(
        MockTokenizer(vocab_map, **t_kwargs)
    )
    client.model = mock_model
    return client


class MockTokenizer:
    def __init__(self, vocab_map, model_max_length=512, return_mask=False):
        self.model_max_length = model_max_length
        self.returns_mask = return_mask
        self.vocab = vocab_map
        self.id_to_token = {id: token for token, id in self.vocab.items()}

    def __call__(self, text: Union[str, List[str]], **kwargs) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]
        token_ids = []
        mask = []
        for sequence in text:
            sequence_ids = [self.vocab.get(token, 0) for token in sequence.split()]
            token_ids.append(sequence_ids)
            mask.append([1] * len(sequence_ids))

        encoding_data = {"input_ids": token_ids}
        if self.returns_mask:
            encoding_data["attention_mask"] = mask

        return BatchEncoding(encoding_data, tensor_type=kwargs.get("return_tensors"))

    def decode(self, token_ids: Union[int, List[int], "torch.Tensor"]) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()  # type: ignore
        decoded_tokens = self.convert_ids_to_tokens(token_ids)  # type: ignore
        if isinstance(decoded_tokens, list):
            return " ".join(decoded_tokens)
        return decoded_tokens

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.id_to_token[ids]
        return [self.id_to_token[id] for id in ids]
