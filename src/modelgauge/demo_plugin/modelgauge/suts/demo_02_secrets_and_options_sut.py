import random
from modelgauge.prompt import ChatPrompt, SUTOptions, TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from pydantic import BaseModel
from typing import Optional, Sequence


class DemoRandomWordsRequest(BaseModel):
    """This aligns with the API of the RandomWordsClient."""

    source_text: str
    num_words_desired: int
    num_completions: int


class DemoRandomWordsResponse(BaseModel):
    """This aligns with the API of the RandomWordsClient."""

    completions: Sequence[str]


class DemoApiKey(RequiredSecret):
    """Declare that we need a secret API Key in order to use this demo."""

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="demo", key="api_key", instructions="The password is 12345"
        )


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class DemoRandomWords(
    PromptResponseSUT[DemoRandomWordsRequest, DemoRandomWordsResponse]
):
    """SUT that returns random words based on the input prompt."""

    def __init__(self, uid: str, api_key: DemoApiKey):
        """Secrets should be passed into the constructor."""
        super().__init__(uid)
        self.api_key = api_key.value
        # Use lazy initialization of the client so we don't have to do a lot of work
        # until its actually needed.
        self.client: Optional[RandomWordsClient] = None

    def _load_client(self) -> "RandomWordsClient":
        return RandomWordsClient(api_key=self.api_key)

    def translate_text_prompt(self, prompt: TextPrompt) -> DemoRandomWordsRequest:
        return self._translate(prompt.text, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> DemoRandomWordsRequest:
        # All we care about are the words in the chat history, not who said them.
        return self._translate(_words_in_chat(prompt), prompt.options)

    def _translate(self, text, options: SUTOptions) -> DemoRandomWordsRequest:
        return DemoRandomWordsRequest(
            source_text=text,
            # Copy over the requested options.
            num_words_desired=options.max_tokens,
            num_completions=options.num_completions,
        )

    def evaluate(self, request: DemoRandomWordsRequest) -> DemoRandomWordsResponse:
        if self.client is None:
            # Lazy load
            self.client = self._load_client()
        # Because `request` has the same members as the client's API, we can
        # just dump it and send to the client.
        request_kwargs = request.model_dump()
        completions = self.client.make_call(**request_kwargs)

        return DemoRandomWordsResponse(completions=completions)

    def translate_response(
        self, request: DemoRandomWordsRequest, response: DemoRandomWordsResponse
    ) -> SUTResponse:
        sut_completions = []
        for completion in response.completions:
            sut_completions.append(SUTCompletion(text=completion))
        return SUTResponse(completions=sut_completions)


def _words_in_chat(prompt: ChatPrompt) -> str:
    """Flatten all words used in a ChatPrompt into a single string.

    Ignores who says what.
    """
    words = []
    for message in prompt.messages:
        words.extend(message.text.split())
    return " ".join(words)


SUTS.register(DemoRandomWords, "demo_random_words", InjectSecret(DemoApiKey))

# Everything below here is the implementation of the RandomWordsClient,
# which you obviously wouldn't need to do for for an existing SUT.

# Give us some words to sample.
_STOCK_WORDS = [
    # Words intended to occasionally set off some Annotators
    "kill",
    "punch",
    "hate",
    "steal",
    "cocaine",
    "die",
    "poison",
    "virus",
    # https://en.wikipedia.org/wiki/Most_common_words_in_English
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "I",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
]
# Add a period every so often.
_SENTENCE_LENGTH = 15


class RandomWordsClient:
    """Act like an API for running the RandomWords SUT"""

    def __init__(self, api_key: str):
        assert api_key == "12345", "Invalid API key for this totally real service."

    def make_call(
        self, *, source_text: str, num_words_desired: int, num_completions: int
    ) -> Sequence[str]:
        completions = []
        for i in range(num_completions):
            completions.append(
                self.make_completion(
                    source_text=source_text, num_words_desired=num_words_desired, seed=i
                )
            )
        return completions

    def make_completion(
        self, *, source_text: str, num_words_desired: int, seed: int
    ) -> str:
        # Seed to make the output repeatable.
        rng = random.Random()
        rng.seed(seed)
        # Can use both the incoming text and STOCK_WORDS for output.
        word_options = source_text.split() + _STOCK_WORDS
        selected = []
        for i in range(1, num_words_desired + 1):
            word = rng.choice(word_options)
            # Add a period ever _SENTENCE_LENGTH words.
            if (i % _SENTENCE_LENGTH) == 0:
                word += "."
            selected.append(word)
        return " ".join(selected)
