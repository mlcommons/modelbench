"""
This module supports the common pattern of structuring Prompts
as multiple choice questions.
"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MultipleChoiceQuestion:
    """Hold all the data relevant to a single question."""

    question: str
    options: List[str]
    correct_option: int  # Index into options


@dataclass(frozen=True, kw_only=True)
class MultipleChoiceFormatting:
    """Options for how to convert MultipleChoiceQuestion into Prompt text."""

    question_prefix: str = ""
    question_suffix: str = ""
    option_identifiers: List[str]
    option_identifier_separator: str
    option_separator: str
    answer_prefix: str
    answer_suffix: str = ""
    # Used when combining multiple questions together.
    instructions_block: str = ""
    block_separator: str = ""


def basic_multiple_choice_format() -> MultipleChoiceFormatting:
    """A standard setting for how to format multiple choice questions."""
    return MultipleChoiceFormatting(
        question_prefix="Question: ",
        question_suffix="\n",
        # All capital letters
        option_identifiers=[chr(ord("A") + i) for i in range(26)],
        option_identifier_separator=") ",
        option_separator="\n",
        answer_prefix="\nAnswer: ",
        answer_suffix="\n",
        instructions_block="The following are multiple choice questions (with answers).",
        block_separator="\n",
    )


def question_with_training_to_text(
    eval_question: MultipleChoiceQuestion,
    training_questions: List[MultipleChoiceQuestion],
    formatting: MultipleChoiceFormatting,
) -> str:
    """Creates a series of multiple choice question blocks."""

    blocks: List[str] = []
    if formatting.instructions_block:
        blocks.append(formatting.instructions_block)

    # Text for in-context training questions
    for train in training_questions:
        blocks.append(question_to_text(train, formatting, include_answer=True))

    # The eval question's text
    blocks.append(question_to_text(eval_question, formatting, include_answer=False))

    return formatting.block_separator.join(blocks)


def question_to_text(
    question: MultipleChoiceQuestion,
    formatting: MultipleChoiceFormatting,
    include_answer: bool,
) -> str:
    """Formats a multiple choice question."""
    # Start with the question
    result: str = (
        formatting.question_prefix + question.question + formatting.question_suffix
    )

    # Add each option
    option_blocks = []
    for option_index, option in enumerate(question.options):
        identifier = formatting.option_identifiers[option_index]
        option_blocks.append(
            identifier + formatting.option_identifier_separator + option
        )

    result += formatting.option_separator.join(option_blocks)

    # Either include the answer or the prefix to the answer.
    if include_answer:
        correct_identifier = formatting.option_identifiers[question.correct_option]
        result += (
            formatting.answer_prefix + correct_identifier + formatting.answer_suffix
        )
    else:
        result += formatting.answer_prefix.rstrip()

    return result
