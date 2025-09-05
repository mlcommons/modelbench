from abc import ABC, abstractmethod

import tiktoken


class Tokenizer(ABC):
    def __init__(self):
        self._encoding = None

    @property
    def encoding(self):
        if self._encoding is None:
            self._encoding = self._get_encoding()
        return self._encoding

    @abstractmethod
    def _get_encoding(self):
        pass

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.encoding.decode(tokens)
        return text


class GeneralTokenizer(Tokenizer):
    def _get_encoding(self):
        return tiktoken.get_encoding("cl100k_base")
