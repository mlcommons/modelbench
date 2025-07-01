import pytest
from pathlib import Path
from typing import Iterable

from modelgauge.data_schema import (
    DEFAULT_PROMPT_RESPONSE_SCHEMA,
    DEFAULT_PROMPT_SCHEMA,
    SchemaValidationError,
)
from modelgauge.dataset import (
    AnnotationDataset,
    BaseDataset,
    PromptDataset,
    PromptResponseDataset,
    SutInteraction,
)
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse


class TestBaseDataset:
    """Tests for the base dataset functionality."""

    class DummySchema:
        def __init__(self):
            self.header = ["col_a"]
            self.col_a = "col_a"

    class DummyDataset(BaseDataset):
        def __init__(self, path: Path, mode: str):
            super().__init__(path, mode)
            self.row_to_item_called = False
            self.write_called = False
            self.write_item = None

        def _init_schema(self):
            self.schema = TestBaseDataset.DummySchema()

        def row_to_item(self, row: dict) -> str:
            """Convert a row to a dummy item."""
            self.row_to_item_called = True
            return row[self.schema.col_a]

        def item_to_row(self, item: str) -> list[str]:
            """Write a dummy item."""
            self.write_called = True
            self.write_item = item
            return [item]

    @pytest.fixture
    def dummy_csv(self, tmp_path):
        """Create a dummy CSV file."""
        file_path = tmp_path / "dummy.csv"
        file_path.write_text("col_a\ndata1")
        return file_path

    @pytest.fixture
    def dummy_read_dataset(self, dummy_csv):
        return self.DummyDataset(dummy_csv, mode="r")

    @pytest.fixture
    def dummy_write_dataset(self, tmp_path):
        return self.DummyDataset(tmp_path / "dummy.csv", mode="w")

    def test_invalid_mode(self, tmp_path):
        with pytest.raises(AssertionError, match="Invalid dataset mode"):
            self.DummyDataset(tmp_path / "data.csv", mode="x")

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.DummyDataset("nonexistent.csv", mode="r")

    def test_cannot_use_existing_file_for_write(self, dummy_csv):
        with pytest.raises(FileExistsError):
            self.DummyDataset(dummy_csv, mode="w")

    def test_len(self, dummy_read_dataset):
        assert len(dummy_read_dataset) == 1
    
    def test_len_in_write_mode(self, dummy_write_dataset):
        with pytest.raises(NotImplementedError, match="Length not supported in write mode"):
            len(dummy_write_dataset)

    def test_schema_is_set_in_initialization(self, dummy_read_dataset):
        assert isinstance(dummy_read_dataset.schema, self.DummySchema)

    def test_header_columns(self, dummy_read_dataset):
        assert dummy_read_dataset.header_columns() == ["col_a"]

    def test_context_manager_read(self, dummy_read_dataset):
        """Test that context manager properly opens and closes files in read mode."""
        assert dummy_read_dataset.file is None

        with dummy_read_dataset:
            assert dummy_read_dataset.file is not None
            assert not dummy_read_dataset.file.closed
            assert dummy_read_dataset.reader is not None

        assert dummy_read_dataset.file is None
        assert dummy_read_dataset.reader is None

    def test_context_manager_write(self, dummy_write_dataset):
        """Test that context manager properly opens and closes files in write mode."""
        assert dummy_write_dataset.file is None

        with dummy_write_dataset:
            assert dummy_write_dataset.file is not None
            assert not dummy_write_dataset.file.closed
            assert dummy_write_dataset.writer is not None

        assert dummy_write_dataset.file is None
        assert dummy_write_dataset.writer is None

    def test_read_in_write_mode(self, dummy_write_dataset):
        """Test that reading in write mode raises an error."""
        with pytest.raises(RuntimeError, match="Can only iterate over dataset in read mode."):
            for _ in dummy_write_dataset:
                break  # Iteration forces read.

    def test_write_in_read_mode(self, dummy_read_dataset):
        """Test that writing in read mode raises an error."""
        with pytest.raises(RuntimeError, match="Cannot write to dataset in read mode"):
            with dummy_read_dataset:
                dummy_read_dataset.write("test")

    def test_row_to_item_called(self, dummy_read_dataset):
        """Test that row_to_item is called when iterating."""
        assert dummy_read_dataset.row_to_item_called is False
        with dummy_read_dataset:
            for obj in dummy_read_dataset:
                assert dummy_read_dataset.row_to_item_called is True
                assert obj == "data1"

    def test_write_operation(self, dummy_write_dataset):
        """Test that write operation works in write mode."""
        with dummy_write_dataset:
            dummy_write_dataset.write("test_data")
            assert dummy_write_dataset.write_called is True
            assert dummy_write_dataset.write_item == "test_data"

    def test_header_gets_written_to_new_file(self, dummy_write_dataset):
        """Test that header is written to a new file."""
        with dummy_write_dataset:
            pass
        assert dummy_write_dataset.path.read_text() == "col_a\n"

    def test_append_to_existing_file(self, dummy_write_dataset):
        """Test that header does not get re-written if context manager is entered twice."""
        with dummy_write_dataset:
            pass
        with dummy_write_dataset:
            pass
        assert dummy_write_dataset.path.read_text() == "col_a\n"

    def test_iteration_enters_exits_context(self, dummy_read_dataset):
        """Test that iteration enters and exits the context."""
        assert dummy_read_dataset.file is None
        for obj in dummy_read_dataset:
            assert dummy_read_dataset.file is not None
            assert not dummy_read_dataset.file.closed
            assert obj == "data1"
            break
        assert dummy_read_dataset.file is None


class TestPromptDataset:
    @pytest.fixture
    def sample_prompts_csv(self, tmp_path):
        """Create a sample CSV file with prompts only."""
        file_path = tmp_path / "prompts.csv"
        content = (
            f"{DEFAULT_PROMPT_SCHEMA.prompt_uid},{DEFAULT_PROMPT_SCHEMA.prompt_text}\n"
            "p1,Say hello\n"
            "p2,Say goodbye"
        )
        file_path.write_text(content)
        return file_path

    @pytest.fixture
    def prompts_dataset(self, sample_prompts_csv):
        return PromptDataset(sample_prompts_csv)

    def test_header_columns(self, prompts_dataset):
        assert prompts_dataset.header_columns() == DEFAULT_PROMPT_SCHEMA.header

    def test_iterate_explicit_context(self, prompts_dataset):
        with prompts_dataset as dataset:
            items = []
            for item in dataset:
                items.append(item)
            assert len(items) == 2
            assert all(isinstance(item, TestItem) for item in items)

            assert items[0].source_id == "p1"
            assert items[0].prompt.text == "Say hello"

            assert items[1].source_id == "p2"
            assert items[1].prompt.text == "Say goodbye"

    def test_iterate_implicit_context(self, prompts_dataset):
        items = list(prompts_dataset)
        assert len(items) == 2
        assert all(isinstance(item, TestItem) for item in items)

    def test_invalid_schema(self, tmp_path):
        """Test that reading a CSV file with invalid schema raises an error."""
        file_path = tmp_path / "invalid.csv"
        file_path.write_text("column1,column2\na,b\n")

        with pytest.raises(SchemaValidationError):
            PromptDataset(file_path)


class TestPromptResponseDataset:
    @pytest.fixture
    def sample_responses_csv(self, tmp_path):
        """Create a sample CSV file with prompt-response data."""
        file_path = tmp_path / "responses.csv"
        content = (
            f"{DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid},{DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text},"
            f"{DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid},{DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response}\n"
            "p1,Say hello,sut1,Hello world\n"
            "p2,Say goodbye,sut1,Goodbye world"
        )
        file_path.write_text(content)
        return file_path

    def test_schema_read(self, sample_responses_csv):
        dataset = PromptResponseDataset(sample_responses_csv, mode="r")
        assert dataset.schema.header == DEFAULT_PROMPT_RESPONSE_SCHEMA.header

    def test_schema_write(self, tmp_path):
        dataset = PromptResponseDataset(tmp_path / "responses.csv", mode="w")
        assert dataset.schema == DEFAULT_PROMPT_RESPONSE_SCHEMA

    def test_read_csv(self, sample_responses_csv):
        with PromptResponseDataset(sample_responses_csv, mode="r") as dataset:
            interactions = list(dataset)
            assert len(interactions) == 2
            assert all(isinstance(interaction, SutInteraction) for interaction in interactions)

            # Check first interaction
            assert interactions[0].prompt.source_id == "p1"
            assert interactions[0].prompt.prompt.text == "Say hello"
            assert interactions[0].sut_uid == "sut1"
            assert interactions[0].response.text == "Hello world"

            # Check second interaction
            assert interactions[1].prompt.source_id == "p2"
            assert interactions[1].prompt.prompt.text == "Say goodbye"
            assert interactions[1].sut_uid == "sut1"
            assert interactions[1].response.text == "Goodbye world"

    def test_write_csv(self, tmp_path):
        output_file = tmp_path / "output.csv"

        # Create test data
        interaction = SutInteraction(
            prompt=TestItem(prompt=TextPrompt(text="Test prompt"), source_id="test1", context={}),
            sut_uid="sut1",
            response=SUTResponse(text="Test response"),
        )

        # Write data
        with PromptResponseDataset(output_file, mode="w") as dataset:
            dataset.write(interaction)

        # Verify written data
        assert output_file.exists()
        content = output_file.read_text()
        expected_header = (
            f"{DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid},{DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text},"
            f"{DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid},{DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response}\n"
        )
        expected_data = "test1,Test prompt,sut1,Test response\n"
        assert content == expected_header + expected_data


# class TestAnnotationDataset:
# @pytest.fixture
# def sample_annotations_jsonl(tmp_path):
#     """Create a sample JSONL file with annotated prompt-response data."""
#     file_path = tmp_path / "annotations.jsonl"
#     content = [
#         {
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid: "p1",
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text: "Say hello",
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid: "sut1",
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response: "Hello world",
#             "Annotations": {"toxicity": 0.1}
#         },
#         {
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid: "p2",
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text: "Say goodbye",
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid: "sut1",
#             DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response: "Goodbye world",
#             "Annotations": {"toxicity": 0.0}
#         }
#     ]
#     import jsonlines
#     with jsonlines.open(file_path, mode='w') as writer:
#         writer.write_all(content)
#     return file_path
#
#     def test_read_jsonl(self, sample_annotations_jsonl):
#         """Test reading annotated prompt-response pairs from a JSONL file."""
#         with AnnotationDataset(sample_annotations_jsonl, mode='r') as dataset:
#             interactions = list(dataset)
#             assert len(interactions) == 2

#             # Check first interaction
#             interaction, annotations = interactions[0]
#             assert interaction.prompt.source_id == "p1"
#             assert interaction.prompt.prompt.text == "Say hello"
#             assert interaction.sut_uid == "sut1"
#             assert interaction.response.text == "Hello world"
#             assert annotations == {"toxicity": 0.1}

#             # Check second interaction
#             interaction, annotations = interactions[1]
#             assert interaction.prompt.source_id == "p2"
#             assert annotations == {"toxicity": 0.0}

#     def test_read_csv(self, sample_responses_csv):
#         """Test reading from a CSV file (should have no annotations)."""
#         with AnnotationDataset(sample_responses_csv, mode='r') as dataset:
#             interactions = list(dataset)
#             assert len(interactions) == 2

#             # Check that annotations are None
#             for interaction, annotations in interactions:
#                 assert annotations is None

#     def test_write_jsonl(self, tmp_path):
#         """Test writing annotated prompt-response pairs to a JSONL file."""
#         output_file = tmp_path / "output.jsonl"

#         # Create test data
#         interaction = SutInteraction(
#             prompt=TestItem(
#                 prompt=TextPrompt(text="Test prompt"),
#                 source_id="test1",
#                 context={}
#             ),
#             sut_uid="sut1",
#             response=SUTResponse(text="Test response")
#         )
#         annotations = {"toxicity": 0.5}

#         # Write data
#         with AnnotationDataset(output_file, mode='w') as dataset:
#             dataset.write(interaction, annotations)

#         # Verify written data
#         import jsonlines
#         with jsonlines.open(output_file) as reader:
#             data = list(reader)
#             assert len(data) == 1
#             assert data[0][DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid] == "test1"
#             assert data[0][DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text] == "Test prompt"
#             assert data[0][DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid] == "sut1"
#             assert data[0][DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response] == "Test response"
#             assert data[0]["Annotations"] == {"toxicity": 0.5}

#     def test_write_csv(self, tmp_path):
#         """Test writing to a CSV file (should ignore annotations)."""
#         output_file = tmp_path / "output.csv"

#         # Create test data
#         interaction = SutInteraction(
#             prompt=TestItem(
#                 prompt=TextPrompt(text="Test prompt"),
#                 source_id="test1",
#                 context={}
#             ),
#             sut_uid="sut1",
#             response=SUTResponse(text="Test response")
#         )
#         annotations = {"toxicity": 0.5}

#         # Write data
#         with AnnotationDataset(output_file, mode='w') as dataset:
#             dataset.write(interaction, annotations)

#         # Verify written data - should not include annotations
#         assert output_file.exists()
#         content = output_file.read_text()
#         expected_header = f"{DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid},{DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text}," \
#                          f"{DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid},{DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response}\n"
#         expected_data = '"test1","Test prompt","sut1","Test response"\n'
#         assert content == expected_header + expected_data
