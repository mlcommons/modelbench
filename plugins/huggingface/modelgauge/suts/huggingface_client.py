import logging
import os
import torch
from copy import deepcopy
from modelgauge.concurrency import ThreadSafeWrapper
from modelgauge.general import value_or_default
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.secret_values import InjectSecret, OptionalSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from pydantic import BaseModel
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
)
from transformers.generation.stopping_criteria import (  # type: ignore
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Any, Dict, List, Optional, Tuple

WrappedPreTrainedTokenizer = ThreadSafeWrapper[PreTrainedTokenizerBase]
"""Thread safe wrapper around Hugging Face PreTrainedTokenizerBase.

Hugging Face PreTrainedTokenizerBase is thread-hostile and using it from multiple threads
simultaneously can result in an "Already borrowed" error (#1421). This wrapper ensures
that a lock is held when using the PreTrainedTokenizerBase.

Example usage:

    with wrapped_tokenizer as tokenizer:
        tokenizer.encode("...")
"""


def create_tokenizer(
    pretrained_model_name_or_path: str, hugging_face_token: Optional[str], **kwargs
) -> WrappedPreTrainedTokenizer:
    """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
    # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
    # TODO: Figure out if we actually need this.
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    try:
        # From the Hugging Face documentation, "local_files_only(defaults to False) —
        # Whether or not to only look at local files".
        # Running `local_files_only=False` requires an internet connection even if the files are downloaded
        # and cached. We need to first run with `local_files_only=True` just in case the machine
        # we are running this code has connection issues. If the tokenizer files are not cached,
        # we attempt to download them from HuggingFace.
        # From https://huggingface.co/course/chapter6/3, "slow tokenizers are those written in Python inside
        # the Hugging Face Transformers library, while the fast versions are the ones provided by Hugging Face
        # Tokenizers, which are written in Rust." So, use the "fast" version of the tokenizers if available.
        return WrappedPreTrainedTokenizer(
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=True,
                use_fast=True,
                token=hugging_face_token,
                **kwargs,
            )
        )
    except OSError:
        return WrappedPreTrainedTokenizer(
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=False,
                use_fast=True,
                token=hugging_face_token,
                **kwargs,
            )
        )


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence: List[int]):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(
            self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype
        )

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())


class HuggingFaceRequest(BaseModel):
    """Data passed between make_request and serve_request. Used as the cache key."""

    prompt: str
    model: str  # Included to help uniquely identify the request, for example in caching.
    temperature: float
    num_return_sequences: int
    max_new_tokens: int
    top_p: float
    top_k_per_token: int
    stop_sequences: List


class HuggingFaceCompletion(BaseModel):
    text: str
    tokens: List[str]
    logprobs: List[float]
    top_logprobs_dicts: List[Dict[str, float]]


class HuggingFaceResponse(BaseModel):
    completions: List[HuggingFaceCompletion]
    input_length: int


class Token(BaseModel):
    """
    A `Token` represents one token position in a `Sequence`, which has the
    chosen `text` as well as the top probabilities under the model.

    Note: (text, logprob) could exist or not exist in `top_logprobs`.
    """

    # Text that was chosen
    text: str

    # Log probability of generating that
    logprob: float

    # text -> log probability of generating that
    top_logprobs: Dict[str, float]


class Sequence(BaseModel):
    """A `Sequence` is a sequence of tokens."""

    # The concatenation of all the tokens
    text: str

    # The sum of the log probabilities of all tokens
    logprob: float

    # The tokens
    tokens: List[Token]


def _truncate_sequence(
    sequence: Sequence, request: HuggingFaceRequest, print_warning: bool = True
) -> Sequence:
    """
    Certain providers have bugs where they aren't respecting max_tokens,
    stop_sequences and the end of text token, so as a hack, we have to manually
    truncate the suffix of `sequence` and `tokens` as a post-hoc process.
    """

    for stop in request.stop_sequences:
        # Find `stop` in the text
        try:
            new_text = sequence.text[: sequence.text.index(stop)]
        except ValueError:
            # The stop sequence doesn't exist, but it might exist in the list of tokens.
            new_text = sequence.text

        # Strip `stop` off the tokens
        new_tokens: List[Token] = []
        # Need to start
        for token in sequence.tokens:
            # Note: we can only strip at token boundaries
            if token.text.startswith(stop):
                break
            new_tokens.append(token)

        # Recompute log probability
        new_logprob = sum(token.logprob for token in new_tokens)
        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    # Truncate based on the max number of tokens.
    if len(sequence.tokens) > request.max_new_tokens:
        new_tokens = sequence.tokens[: request.max_new_tokens]

        # This is imperfect stitching together of tokens, so just to make sure this is okay
        # TODO: should use the proper detokenizer since T5-style models.
        # Usually, in our benchmark, max_tokens is active when it's 1, so hopefully this isn't an issue.
        new_text = "".join(token.text for token in new_tokens)
        new_logprob = sum(token.logprob for token in new_tokens)

        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    return sequence


TORCH_DTYPE_KEY = "torch_dtype"
TORCH_DTYPE_VALUE_PREFIX = "torch."


def _process_huggingface_client_kwargs(raw_kwargs: Dict[str, Any]):
    """Process the kwargs for HuggingFaceClient.

    The kwargs passed to HuggingFaceClient will eventually be passed to AutoModel.from_pretrained().
    Since the kwargs from HuggingFaceClient may be derived from configuration YAML,
    they may contain primitive types instead of the unserializable types that
    AutoModel.from_pretrained() expects (e.g. torch_dtype). This function converts values of
    primitive types to values of the unserializable types."""
    processed_kwargs = deepcopy(raw_kwargs)

    # Convert torch_dtype string value to actual dtypes
    # e.g. the string "torch.bfloat16" is converted to torch.bfloat16
    torch_dtype = processed_kwargs.get(TORCH_DTYPE_KEY)
    if torch_dtype and isinstance(torch_dtype, str):
        if not torch_dtype.startswith(TORCH_DTYPE_VALUE_PREFIX):
            raise ValueError(
                f'Unknown dtype "{torch_dtype}"; expected a string such as "torch.bfloat16"'
            )
        processed_kwargs[TORCH_DTYPE_KEY] = getattr(
            torch, torch_dtype[len(TORCH_DTYPE_VALUE_PREFIX) :]
        )

    return processed_kwargs


class HuggingFaceToken(OptionalSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="hugging_face",
            key="token",
            instructions="You can create tokens at https://huggingface.co/settings/tokens.",
        )


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class HuggingFaceSUT(PromptResponseSUT[HuggingFaceRequest, HuggingFaceResponse]):
    """A thin wrapper around a Hugging Face AutoModelForCausalLM for HuggingFaceClient to call."""

    def __init__(
        self,
        uid: str,
        pretrained_model_name_or_path: str,
        token: HuggingFaceToken,
        **kwargs,
    ):
        super().__init__(uid)
        if torch.cuda.is_available():
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        self.token = token.value
        self.hg_kwargs = _process_huggingface_client_kwargs(kwargs)
        self.model_path = pretrained_model_name_or_path
        # Model and tokenizer are lazy loaded.
        self.model: Optional[Any] = None
        self.wrapped_tokenizer: Optional[WrappedPreTrainedTokenizer] = None

    def _load_model(self) -> Tuple[Any, WrappedPreTrainedTokenizer]:
        # WARNING this may fail if your GPU does not have enough memory
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            token=self.token,
            **self.hg_kwargs,
        ).to(self.device)
        wrapped_tokenizer = create_tokenizer(
            self.model_path, self.token, **self.hg_kwargs
        )
        return model, wrapped_tokenizer

    def evaluate(self, raw_request: HuggingFaceRequest) -> HuggingFaceResponse:
        assert self.model_path == raw_request.model
        if not self.model or not self.wrapped_tokenizer:
            self.model, self.wrapped_tokenizer = self._load_model()
        with self.wrapped_tokenizer as tokenizer:
            encoded_input = tokenizer(
                raw_request.prompt, return_tensors="pt", return_token_type_ids=False
            ).to(self.device)
            num_input_tokens = encoded_input.input_ids.nelement()
            # Ensure the total tokens is within the model's max length.
            max_new_tokens = min(
                raw_request.max_new_tokens,
                tokenizer.model_max_length - num_input_tokens,
            )
            if raw_request.max_new_tokens != max_new_tokens:
                logging.warning(
                    f"Had to reduce max_new_tokens from "
                    f"{raw_request.max_new_tokens} to {max_new_tokens}"
                )
            assert max_new_tokens >= 0, (
                f"Prompt has {num_input_tokens}, which is larger than "
                f"max length {tokenizer.model_max_length}"
            )
        top_k_per_token: int = raw_request.top_k_per_token
        stopping_criteria: Optional[StoppingCriteriaList] = None
        optional_args = {}
        if len(raw_request.stop_sequences) > 0:
            with self.wrapped_tokenizer as tokenizer:
                stop_sequence_ids = tokenizer(
                    raw_request.stop_sequences,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )
            if (
                len(stop_sequence_ids.input_ids) == 1
                and len(stop_sequence_ids.input_ids[0]) == 1
            ):
                optional_args["eos_token_id"] = stop_sequence_ids.input_ids[0][0]
            else:
                stopping_criteria = StoppingCriteriaList()
                for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                    stopping_criteria.append(
                        StopAtSpecificTokenCriteria(
                            stop_sequence=stop_sequence_input_ids
                        )
                    )

        # TODO: Dedent this after refactor
        if True:
            # Some models do not have a `pad_token_id`. For example gpt2
            # This prevents a warning message
            generation_config = None
            if self.model.config.pad_token_id is None:
                if self.model.config.eos_token_id is not None:
                    generation_config = GenerationConfig(
                        pad_token_id=self.model.config.eos_token_id
                    )
            output = self.model.generate(
                **encoded_input,
                temperature=raw_request.temperature,
                num_return_sequences=raw_request.num_return_sequences,
                max_new_tokens=max_new_tokens,
                top_p=raw_request.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                generation_config=generation_config,
                **optional_args,
                stopping_criteria=stopping_criteria,
            )
            sequences = output.sequences
            scores = output.scores

        # Compute logprobs of generated tokens for each completed sequence.
        all_generated_tokens_logprobs = []
        all_generated_tokens_top_logprobs_dicts = []
        for completion_id in range(raw_request.num_return_sequences):
            generated_tokens_logprobs = []
            generated_tokens_top_logprobs_dicts = []
            for i in range(
                len(sequences[completion_id]) - len(encoded_input.input_ids[0])
            ):
                logprobs = torch.nn.functional.log_softmax(
                    scores[i][completion_id], dim=0
                )
                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                with self.wrapped_tokenizer as tokenizer:
                    generated_tokens_top_logprobs_dicts.append(
                        {
                            tokenizer.convert_ids_to_tokens(k.item()): v.item()
                            for (k, v) in zip(
                                topk_logprobs.indices, topk_logprobs.values
                            )
                        }
                    )
                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                generated_tokens_logprobs.append(
                    logprobs[sequences[completion_id][j]].item()
                )
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)
            all_generated_tokens_top_logprobs_dicts.append(
                generated_tokens_top_logprobs_dicts
            )

        # Remove prompt from the start of each sequence.
        sequences = [
            sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences
        ]

        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [
                [tokenizer.decode(token) for token in sequence_tokens]
                for sequence_tokens in sequences
            ]
            all_decoded_text = tokenizer.batch_decode(sequences)

        completions: List[HuggingFaceCompletion] = []
        for (
            decoded_text,
            tokens,
            generated_tokens_logprobs,
            generated_tokens_top_logprobs_dicts,
        ) in zip(
            all_decoded_text,
            all_tokens,
            all_generated_tokens_logprobs,
            all_generated_tokens_top_logprobs_dicts,
        ):
            completions.append(
                HuggingFaceCompletion(
                    text=decoded_text,
                    tokens=tokens,
                    logprobs=generated_tokens_logprobs,
                    top_logprobs_dicts=generated_tokens_top_logprobs_dicts,
                )
            )

        return HuggingFaceResponse(
            completions=completions,
            input_length=len(encoded_input.input_ids[0]),
        )

    def translate_text_prompt(self, prompt: TextPrompt) -> HuggingFaceRequest:
        return self._translate_request(prompt.text, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> HuggingFaceRequest:
        return self._translate_request(format_chat(prompt), prompt.options)

    def _translate_request(self, text, options):
        request = {
            "model": self.model_path,
            "prompt": text,
            "max_new_tokens": options.max_tokens,
            "num_return_sequences": options.num_completions,
        }

        # Handle defaulting
        temperature = value_or_default(options.temperature, 1.0)
        if temperature == 0:
            temperature = 1e-7
        request["temperature"] = temperature
        request["top_p"] = value_or_default(options.top_p, 1)
        request["top_k_per_token"] = value_or_default(options.top_k_per_token, 1)
        request["stop_sequences"] = value_or_default(options.stop_sequences, [])
        return HuggingFaceRequest.model_validate(request)

    def translate_response(
        self, request: HuggingFaceRequest, response: HuggingFaceResponse
    ) -> SUTResponse:
        completions = []
        for raw_completion in response.completions:
            sequence_logprob: float = 0
            tokens: List[Token] = []
            generated_tokens = raw_completion.tokens

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens,
                raw_completion.logprobs,
                raw_completion.top_logprobs_dicts,
            ):
                tokens.append(
                    Token(
                        text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict
                    )
                )
                sequence_logprob += logprob

            completion = Sequence(
                text=raw_completion.text, logprob=sequence_logprob, tokens=tokens
            )
            completion = _truncate_sequence(completion, request)
            completions.append(completion)
        sut_completions = [SUTCompletion(text=c.text) for c in completions]
        return SUTResponse(completions=sut_completions)


SUTS.register(HuggingFaceSUT, "gpt2", "gpt2", InjectSecret(HuggingFaceToken))
