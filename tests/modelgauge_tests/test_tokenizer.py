import pytest

from modelgauge.tokenizer import GeneralTokenizer, Tokenizer


class SimpleTokenizer(Tokenizer):
    class SimpleEncoding:
        def encode(self, text):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    def _get_encoding(self):
        return self.SimpleEncoding()


@pytest.mark.parametrize("text,result", [("One two three", "One two"), ("one", "one")])
def test_simple_tokenizer_truncate(text, result):
    tokenizer = SimpleTokenizer()
    truncated = tokenizer.truncate(text, 2)
    assert truncated == result


def test_general_tokenizer():
    tokenizer = GeneralTokenizer()
    text = "one two three four five"
    truncated = tokenizer.truncate(text, 3)
    assert truncated == "one two three"
