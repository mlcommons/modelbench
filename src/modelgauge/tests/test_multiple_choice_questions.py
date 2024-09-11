from modelgauge.multiple_choice_questions import (
    MultipleChoiceFormatting,
    MultipleChoiceQuestion,
    basic_multiple_choice_format,
    question_to_text,
    question_with_training_to_text,
)


def test_question_to_text_basic_with_answer():
    formatting = basic_multiple_choice_format()
    question = MultipleChoiceQuestion(
        question="What color is the sky?",
        options=["Red", "Green", "Blue"],
        correct_option=2,
    )
    text = question_to_text(question, formatting, include_answer=True)
    assert (
        text
        == """\
Question: What color is the sky?
A) Red
B) Green
C) Blue
Answer: C
"""
    )


def test_question_to_text_basic_without_answer():
    formatting = basic_multiple_choice_format()
    question = MultipleChoiceQuestion(
        question="What color is the sky?",
        options=["Red", "Green", "Blue"],
        correct_option=2,
    )
    text = question_to_text(question, formatting, include_answer=False)
    # No whitespace after "Answer:"
    assert (
        text
        == """\
Question: What color is the sky?
A) Red
B) Green
C) Blue
Answer:"""
    )


def test_question_to_text_alternate_formatting():
    formatting = MultipleChoiceFormatting(
        question_prefix="",
        question_suffix=" ",
        option_identifiers=[str(i + 1) for i in range(3)],
        option_identifier_separator=" - ",
        option_separator=" ",
        answer_prefix=". It is ",
        answer_suffix=".",
    )
    question = MultipleChoiceQuestion(
        question="What color is the sky?",
        options=["Red", "Green", "Blue"],
        correct_option=2,
    )
    text = question_to_text(question, formatting, include_answer=True)
    assert text == """What color is the sky? 1 - Red 2 - Green 3 - Blue. It is 3."""


def test_question_with_training_to_text_basic():
    formatting = basic_multiple_choice_format()
    eval_question = MultipleChoiceQuestion(
        question="What color is the sky?",
        options=["Red", "Green", "Blue"],
        correct_option=2,
    )
    training_1 = MultipleChoiceQuestion(
        question="What goes up",
        options=["Keeps going", "Must come down"],
        correct_option=1,
    )
    training_2 = MultipleChoiceQuestion(
        question="The cow says",
        options=["Moo", "Oink", "Baa", "Hello"],
        correct_option=0,
    )
    text = question_with_training_to_text(
        eval_question, [training_1, training_2], formatting
    )
    assert (
        text
        == """\
The following are multiple choice questions (with answers).
Question: What goes up
A) Keeps going
B) Must come down
Answer: B

Question: The cow says
A) Moo
B) Oink
C) Baa
D) Hello
Answer: A

Question: What color is the sky?
A) Red
B) Green
C) Blue
Answer:"""
    )
