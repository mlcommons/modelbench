import re

from modelgauge.annotators.composer.context import EvalContext
from modelgauge.annotators.composer.nodes import NodeOutput

from evaluators.components.sentence_enricher import Sentence, SentenceEnricher


ASSIGN_RE = re.compile(r"self\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^#\n]+)")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
COMMENT_RE = re.compile(r"#\s*([^\n]+)")
PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
PY_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.M)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
STRING_RE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.S)


class SentenceSegmenter(SentenceEnricher):
    _MAX_SEGMENT_CHARS = 420
    _SEGMENT_STRIDE = 210
    # Match the authority's gate (segment.py:30): four markers, or four-plus
    # lines that are >35% indented.
    _CODE_MARK_THRESHOLD = 4
    _INDENTATION_MIN_LINES = 4
    _INDENTATION_RATIO_THRESHOLD = 0.35
    _CODE_MARKERS = (
        "{",
        "}",
        ";",
        "=>",
        "def ",
        "function ",
        "import ",
        "class ",
    )

    def run(self, ctx: EvalContext) -> NodeOutput:
        sentences = SentenceSegmenter.segment_text(
            ctx.response, max_chars=self._MAX_SEGMENT_CHARS, stride=self._SEGMENT_STRIDE
        )
        updated_ctx = self._update_ctx_sentences(ctx, sentences)
        return NodeOutput(
            value=sentences,
            original_ctx=ctx,
            updated_ctx=updated_ctx,
        )

    @classmethod
    def is_probable_code(cls, text: str) -> bool:
        if "```" in text:
            return True
        code_marks = sum(text.count(mark) for mark in cls._CODE_MARKERS)
        lines = [line for line in text.splitlines() if line.strip()]
        indented = sum(1 for line in lines if line.startswith(("    ", "\t")))
        return code_marks >= cls._CODE_MARK_THRESHOLD or (
            len(lines) >= cls._INDENTATION_MIN_LINES
            and indented / len(lines) > cls._INDENTATION_RATIO_THRESHOLD
        )

    @staticmethod
    def identifier_to_words(identifier: str) -> str:
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", identifier)
        spaced = spaced.replace("_", " ")
        return re.sub(r"\s+", " ", spaced).strip().lower()

    @staticmethod
    def literal_summary(value: str) -> str:
        value = value.strip()
        if value in {'""', "''"}:
            return "empty text"
        if value in {"[]", "list()"}:
            return "empty list"
        if value in {"{}", "dict()"}:
            return "empty dictionary"
        if value in {"None", "null"}:
            return "nothing"
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return f"text {value[1:-1]}"
        return value

    @staticmethod
    def code_to_english_segments(text: str) -> list[Sentence]:
        sentences = []
        seen = set()

        def add(sentence: str, start: int, end: int) -> None:
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence or sentence in seen:
                return
            seen.add(sentence)
            sentences.append(Sentence(sentence, start, end))

        for match in PY_CLASS_RE.finditer(text):
            add(
                f"Class: {SentenceSegmenter.identifier_to_words(match.group(1))}.",
                match.start(),
                match.end(),
            )

        for match in PY_DEF_RE.finditer(text):
            name = SentenceSegmenter.identifier_to_words(match.group(1))
            args = [
                SentenceSegmenter.identifier_to_words(arg.split("=")[0].strip())
                for arg in match.group(2).split(",")
                if arg.strip() and arg.strip() != "self"
            ]
            if args:
                add(
                    f"Method: {name}; inputs: {', '.join(args)}.",
                    match.start(),
                    match.end(),
                )
            else:
                add(f"Method: {name}.", match.start(), match.end())

        # Mask quoted strings so a # inside a literal is not treated as a comment.
        comment_scan_text = STRING_RE.sub(lambda m: " " * (m.end() - m.start()), text)
        for match in COMMENT_RE.finditer(comment_scan_text):
            # Positions come from the masked text, but the body is sliced from
            # the original: `\s*` may have consumed a masked literal, so
            # group(1) can start past text the comment actually contains.
            body = text[match.start() + 1 : match.end()]
            add(f"Comment: {body.strip()}", match.start(), match.end())

        for match in ASSIGN_RE.finditer(text):
            field = SentenceSegmenter.identifier_to_words(match.group(1))
            value = SentenceSegmenter.literal_summary(match.group(2))
            add(f"Stores {field} as {value}.", match.start(), match.end())

        for match in STRING_RE.finditer(text):
            literal = match.group(0)[1:-1].strip()
            if len(literal) >= 4 and re.search(r"[A-Za-z]", literal):
                add(f"String literal: {literal}", match.start(), match.end())

        if sentences:
            return sorted(sentences, key=lambda item: (item.char_start, item.char_end))
        return []

    @staticmethod
    def chunk_text(
        text: str, width: int, stride: int, char_offset: int = 0
    ) -> list[Sentence]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + width)
            raw_chunk = text[start:end]
            chunk = raw_chunk.strip()
            if chunk:
                leading_spaces = len(raw_chunk) - len(raw_chunk.lstrip())
                trailing_spaces = len(raw_chunk) - len(raw_chunk.rstrip())
                chunks.append(
                    Sentence(
                        chunk,
                        char_offset + start + leading_spaces,
                        char_offset + end - trailing_spaces,
                    )
                )
            if end == len(text):
                break
            start += max(1, stride)
        return chunks

    @staticmethod
    def _chunk_code_summary(
        sentence: Sentence, max_chars: int, stride: int
    ) -> list[Sentence]:
        if len(sentence.text) <= max_chars:
            return [sentence]
        return [
            Sentence(chunk.text, sentence.char_start, sentence.char_end)
            for chunk in SentenceSegmenter.chunk_text(sentence.text, max_chars, stride)
        ]

    @staticmethod
    def _segment_plain_text(
        text: str,
        max_chars: int,
        stride: int,
        char_offset: int = 0,
        require_alphanumeric: bool = False,
    ) -> list[Sentence]:
        sentences: list[Sentence] = []
        for line_match in re.finditer(r"[^\n]+", text):
            raw_line = line_match.group(0)
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue

            leading_spaces = len(raw_line) - len(raw_line.lstrip())
            line_start = char_offset + line_match.start() + leading_spaces
            pieces = [line] if BULLET_RE.match(line) else SENTENCE_RE.split(line)
            piece_cursor = 0
            for piece in pieces:
                piece = piece.strip()
                if not piece or (
                    require_alphanumeric and not re.search(r"[A-Za-z0-9]", piece)
                ):
                    continue
                relative_start = line.find(piece, piece_cursor)
                piece_start = line_start + relative_start
                piece_end = piece_start + len(piece)
                piece_cursor = relative_start + len(piece)
                if len(piece) > max_chars:
                    sentences.extend(
                        SentenceSegmenter.chunk_text(
                            piece, max_chars, stride, char_offset=piece_start
                        )
                    )
                else:
                    sentences.append(Sentence(piece, piece_start, piece_end))
        return sentences

    @staticmethod
    def _segment_code_without_losing_text(
        text: str, max_chars: int, stride: int
    ) -> list[Sentence]:
        code_sentences = SentenceSegmenter.code_to_english_segments(text)
        if not code_sentences:
            return SentenceSegmenter._segment_plain_text(text, max_chars, stride)

        sentences: list[Sentence] = []
        cursor = 0
        index = 0
        while index < len(code_sentences):
            group = [code_sentences[index]]
            group_start = group[0].char_start
            group_end = group[0].char_end
            index += 1
            while (
                index < len(code_sentences)
                and code_sentences[index].char_start < group_end
            ):
                group.append(code_sentences[index])
                group_end = max(group_end, code_sentences[index].char_end)
                index += 1

            sentences.extend(
                SentenceSegmenter._segment_plain_text(
                    text[cursor:group_start],
                    max_chars,
                    stride,
                    char_offset=cursor,
                    require_alphanumeric=True,
                )
            )
            for code_sentence in group:
                sentences.extend(
                    SentenceSegmenter._chunk_code_summary(
                        code_sentence, max_chars, stride
                    )
                )
            cursor = group_end

        sentences.extend(
            SentenceSegmenter._segment_plain_text(
                text[cursor:],
                max_chars,
                stride,
                char_offset=cursor,
                require_alphanumeric=True,
            )
        )
        return sorted(sentences, key=lambda item: (item.char_start, item.char_end))

    @staticmethod
    def segment_text(text: str, max_chars: int, stride: int) -> list[Sentence]:
        """Split readable text into model-sized pieces.

        Code is recognised by its ``` fence rather than by counting markers
        across the whole response: markdown headings, semicolon-heavy prose and
        nested bullet lists all trip a marker/indentation heuristic, and a
        mis-routed response has its prose rewritten as synthetic summaries.
        """
        if not text.strip():
            return []

        sentences: list[Sentence] = []
        cursor = 0
        for fence in FENCE_RE.finditer(text):
            if fence.start() > cursor:
                sentences.extend(
                    SentenceSegmenter._segment_plain_text(
                        text[cursor:fence.start()], max_chars, stride, char_offset=cursor
                    )
                )
            sentences.extend(
                SentenceSegmenter._segment_plain_text(
                    fence.group(0), max_chars, stride,
                    char_offset=fence.start(), require_alphanumeric=True,
                )
            )
            cursor = fence.end()
        if cursor < len(text):
            sentences.extend(
                SentenceSegmenter._segment_plain_text(
                    text[cursor:], max_chars, stride, char_offset=cursor
                )
            )
        return sorted(sentences, key=lambda s: (s.char_start, s.char_end))
