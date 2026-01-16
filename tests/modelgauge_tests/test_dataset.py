import csv
import pytest
from pathlib import Path

from pydantic import BaseModel

from modelgauge.data_schema import (
    AnnotationJailbreakSchema,
    AnnotationSchema,
    PromptResponseSchema,
    PromptJailbreakSchema,
    SchemaValidationError,
)
from modelgauge.dataset import (
    AnnotationDataset,
    BaseDataset,
    PromptDataset,
    PromptResponseDataset,
)
from modelgauge.model_options import TopTokens, TokenProbability
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import AnnotatedSUTInteraction, SUTInteraction, TestItem
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

        def _get_schema(self):
            return TestBaseDataset.DummySchema()

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
        file_path.write_text('"col_a"\n"data1"')
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

    def test_invalid_file_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid dataset file"):
            self.DummyDataset(tmp_path / "data.txt", mode="r")

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
        assert dummy_write_dataset.path.read_text() == '"col_a"\n'

    def test_append_to_existing_file(self, dummy_write_dataset):
        """Test that header does not get re-written if context manager is entered twice."""
        with dummy_write_dataset:
            pass
        with dummy_write_dataset:
            pass
        assert dummy_write_dataset.path.read_text() == '"col_a"\n'

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
        schema = PromptJailbreakSchema.default()
        content = (
            f"{schema.prompt_uid},{schema.prompt_text},{schema.evaluated_prompt_text}\n"
            "p1,Say hello,seed 1\n"
            "p2,Say goodbye,seed 2"
        )
        file_path.write_text(content)
        return file_path

    @pytest.fixture
    def prompts_dataset(self, sample_prompts_csv):
        return PromptDataset(sample_prompts_csv)

    @pytest.fixture
    def prompts_jailbreak_dataset(self, sample_prompts_csv):
        return PromptDataset(sample_prompts_csv, jailbreak=True)

    def test_schema_columns_parameterized(self, tmp_path):
        file_path = tmp_path / "different_prompts.csv"
        file_path.write_text('"column1","column2","column3"\n"a","b","c"\n')
        dataset = PromptDataset(
            file_path,
            prompt_uid_col="column1",
            prompt_text_col="column2",
            seed_prompt_text_col="column3",
            jailbreak=True,
        )
        assert dataset.schema.prompt_uid == "column1"
        assert dataset.schema.prompt_text == "column2"
        assert dataset.schema.evaluated_prompt_text == "column3"

    def test_iterate_explicit_context(self, prompts_dataset):
        with prompts_dataset as dataset:
            items = []
            for item in dataset:
                items.append(item)
            assert len(items) == 2
            assert all(isinstance(item, TestItem) for item in items)

            assert items[0].source_id == "p1"
            assert items[0].prompt.text == "Say hello"
            assert items[0].evaluated_prompt == items[0].prompt

            assert items[1].source_id == "p2"
            assert items[1].prompt.text == "Say goodbye"
            assert items[1].evaluated_prompt == items[1].prompt

    def test_iterate_explicit_context_jailbreak(self, prompts_jailbreak_dataset):
        with prompts_jailbreak_dataset as dataset:
            items = []
            for item in dataset:
                items.append(item)
            assert len(items) == 2
            assert all(isinstance(item, TestItem) for item in items)

            assert items[0].source_id == "p1"
            assert items[0].prompt.text == "Say hello"
            assert items[0].evaluated_prompt.text == "seed 1"

            assert items[1].source_id == "p2"
            assert items[1].prompt.text == "Say goodbye"
            assert items[1].evaluated_prompt.text == "seed 2"

    def test_iterate_implicit_context(self, prompts_dataset):
        items = list(prompts_dataset)
        assert len(items) == 2
        assert all(isinstance(item, TestItem) for item in items)

    def test_invalid_schema(self, tmp_path):
        """Test that reading a CSV file with invalid schema raises an error."""
        file_path = tmp_path / "invalid.csv"
        file_path.write_text('"column1","column2"\n"a","b"\n')

        with pytest.raises(SchemaValidationError):
            PromptDataset(file_path)


class TestPromptResponseDataset:
    @pytest.fixture
    def sample_responses_csv(self, tmp_path):
        """Create a sample CSV file with prompt-response data."""
        file_path = tmp_path / "responses.csv"
        schema = PromptResponseSchema.default()
        content = (
            f"{schema.prompt_uid},{schema.prompt_text},"
            f"{schema.sut_uid},{schema.sut_response}\n"
            "p1,Say hello,sut1,Hello world\n"
            "p2,Say goodbye,sut1,Goodbye world"
        )
        file_path.write_text(content)
        return file_path

    def test_schema_read(self, sample_responses_csv):
        dataset = PromptResponseDataset(sample_responses_csv, mode="r")
        assert dataset.schema.header == PromptResponseSchema.default().header

    def test_schema_write(self, tmp_path):
        dataset = PromptResponseDataset(tmp_path / "responses.csv", mode="w")
        assert dataset.schema.header == PromptResponseSchema.default().header

        # Test with logprobs
        dataset = PromptResponseDataset(tmp_path / "responses.csv", mode="w", sut_logprobs=True)
        assert dataset.schema.sut_logprobs == "sut_logprobs"

    def test_schema_columns_parameterized(self, tmp_path):
        file_path = tmp_path / "different_responses.csv"
        file_path.write_text('"c1","c2","c3","c4"\n"a","b"\n')
        dataset = PromptResponseDataset(
            file_path, mode="r", prompt_uid_col="c1", prompt_text_col="c2", sut_uid_col="c3", sut_response_col="c4"
        )
        assert dataset.schema.prompt_uid == "c1"
        assert dataset.schema.prompt_text == "c2"
        assert dataset.schema.sut_uid == "c3"
        assert dataset.schema.sut_response == "c4"

    def test_read_csv(self, sample_responses_csv):
        with PromptResponseDataset(sample_responses_csv, mode="r") as dataset:
            interactions = list(dataset)
            assert len(interactions) == 2
            assert all(isinstance(interaction, SUTInteraction) for interaction in interactions)

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
        interaction = SUTInteraction(
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
        schema = PromptResponseSchema.default()
        expected_header = f"{schema.prompt_uid},{schema.prompt_text}," f"{schema.sut_uid},{schema.sut_response}\n"
        expected_data = "test1,Test prompt,sut1,Test response\n"
        assert content.replace('"', "") == expected_header + expected_data

    def test_write_csv_logprobs(self, tmp_path):
        output_file = tmp_path / "output.csv"

        # Create test data
        logprobs = [
            TopTokens(top_tokens=[TokenProbability(token="Test", logprob=-0.1)]),
            TopTokens(top_tokens=[TokenProbability(token="response", logprob=-0.2)]),
        ]
        interaction = SUTInteraction(
            prompt=TestItem(prompt=TextPrompt(text="Test prompt"), source_id="test1", context={}),
            sut_uid="sut1",
            response=SUTResponse(text="Test response", top_logprobs=logprobs),
        )

        # Write data
        with PromptResponseDataset(output_file, mode="w", sut_logprobs=True) as dataset:
            dataset.write(interaction)

        # Verify written data
        assert output_file.exists()
        content = output_file.read_text()
        schema = PromptResponseSchema.default(sut_logprobs=True)
        expected_header = (
            f"{schema.prompt_uid},{schema.prompt_text},"
            f"{schema.sut_uid},{schema.sut_response},{schema.sut_logprobs}\n"
        )
        expected_logprobs = "[[{'token': 'Test', 'logprob': -0.1}], [{'token': 'response', 'logprob': -0.2}]]"
        expected_data = "test1,Test prompt,sut1,Test response," + expected_logprobs + "\n"
        assert content.replace('"', "") == expected_header + expected_data


class TestAnnotationDataset:
    @pytest.fixture
    def expected_content(self):
        schema = AnnotationSchema.default()
        expected_header = (
            f'"{schema.prompt_uid}","{schema.prompt_text}",'
            f'"{schema.sut_uid}","{schema.sut_response}",'
            f'"{schema.annotator_uid}","{schema.annotation}"\n'
        )
        expected_data = '"test1","Test prompt","sut1","Test response","annotator1","{""is_safe"": true}"\n'
        return expected_header + expected_data

    @pytest.fixture
    def expected_jailbreak_content(self):
        schema = AnnotationJailbreakSchema.default()
        expected_header = (
            f'"{schema.prompt_uid}","{schema.prompt_text}","{schema.evaluated_prompt_text}",'
            f'"{schema.sut_uid}","{schema.sut_response}",'
            f'"{schema.annotator_uid}","{schema.annotation}"\n'
        )
        expected_data = (
            '"test1","Test prompt","Seed prompt","sut1","Test response","annotator1","{""is_safe"": true}"\n'
        )
        return expected_header + expected_data

    @pytest.fixture
    def sample_annotations_csv(self, tmp_path):
        """Sample CSV with QUOTE_ALL quoting."""
        file_path = tmp_path / "annotations.csv"
        schema = AnnotationSchema.default()
        content = (
            f'"{schema.prompt_uid}","{schema.prompt_text}",'
            f'"{schema.sut_uid}","{schema.sut_response}",'
            f'"{schema.annotator_uid}","{schema.annotation}"\n'
            '"p1","Say hello","sut1","Hello world","annotator1","{""is_safe"": true}"\n'
            '"p2","Say goodbye","sut1","Goodbye world","annotator1","{""is_safe"": false}"'
        )
        file_path.write_text(content)
        return file_path

    @pytest.fixture
    def sample_annotations_jailbreak_csv(self, tmp_path):
        """Sample CSV with QUOTE_ALL quoting."""
        file_path = tmp_path / "annotations.csv"
        schema = AnnotationJailbreakSchema.default()
        content = (
            f'"{schema.prompt_uid}","{schema.prompt_text}",'
            f'"{schema.sut_uid}","{schema.sut_response}",'
            f'"{schema.annotator_uid}","{schema.annotation}","{schema.evaluated_prompt_text}"\n'
            '"p1","Say hello","sut1","Hello world","annotator1","{""is_safe"": true}","seed1"\n'
            '"p2","Say goodbye","sut1","Goodbye world","annotator1","{""is_safe"": false}","seed2"'
        )
        file_path.write_text(content)
        return file_path

    def test_schema_read(self, sample_annotations_csv):
        dataset = AnnotationDataset(sample_annotations_csv, mode="r")
        assert dataset.schema.header == AnnotationSchema.default().header

    def test_jailbreak_schema_read(self, sample_annotations_jailbreak_csv):
        dataset = AnnotationDataset(sample_annotations_jailbreak_csv, mode="r", jailbreak=True)
        assert set(dataset.schema.header) == set(AnnotationJailbreakSchema.default().header)

    def test_schema_write(self, tmp_path):
        dataset = AnnotationDataset(tmp_path / "annotations.csv", mode="w")
        assert dataset.schema.header == AnnotationSchema.default().header

        # Test with sut logprobs
        dataset = AnnotationDataset(tmp_path / "annotations.csv", mode="w", sut_logprobs=True)
        assert dataset.schema.sut_logprobs is not None

    def test_jailbreak_schema_write(self, tmp_path):
        dataset = AnnotationDataset(tmp_path / "annotations.csv", mode="w", jailbreak=True)
        assert dataset.schema.header == AnnotationJailbreakSchema.default().header

        # Test with sut logprobs
        dataset = AnnotationDataset(tmp_path / "annotations.csv", mode="w", jailbreak=True, sut_logprobs=True)
        assert dataset.schema.sut_logprobs is not None

    def test_schema_columns_parameterized(self, tmp_path):
        file_path = tmp_path / "different_annotations.csv"
        file_path.write_text('"c1","c2","c3","c4","prompt_uid","prompt_text"\n"a","b"\n')
        dataset = AnnotationDataset(
            file_path, mode="r", annotator_uid_col="c1", annotation_col="c2", sut_uid_col="c3", sut_response_col="c4"
        )
        assert dataset.schema.annotator_uid == "c1"
        assert dataset.schema.annotation == "c2"
        assert dataset.schema.sut_uid == "c3"
        assert dataset.schema.sut_response == "c4"

    def test_jailbreak_schema_columns_parameterized(self, tmp_path):
        file_path = tmp_path / "different_jailbreak_annotations.csv"
        file_path.write_text(
            '"sut_uid","annotator_uid","sut_response","annotation_json","prompt_uid","prompt_text","seed2"\n"a","b"\n'
        )
        dataset = AnnotationDataset(file_path, mode="r", seed_prompt_text_col="seed2", jailbreak=True)
        assert dataset.schema.evaluated_prompt_text == "seed2"

    def test_read_csv(self, sample_annotations_csv):
        with AnnotationDataset(sample_annotations_csv, mode="r") as dataset:
            annotations = list(dataset)
            assert len(annotations) == 2
            assert all(isinstance(annotation, AnnotatedSUTInteraction) for annotation in annotations)

            # Check first annotation
            assert annotations[0].sut_interaction.prompt.source_id == "p1"
            assert annotations[0].sut_interaction.prompt.prompt.text == "Say hello"
            assert annotations[0].sut_interaction.prompt.evaluated_prompt.text == "Say hello"
            assert annotations[0].sut_interaction.sut_uid == "sut1"
            assert annotations[0].sut_interaction.response.text == "Hello world"
            assert annotations[0].annotator_uid == "annotator1"
            assert annotations[0].annotation == {"is_safe": True}  # Annotation jsonstrings are deserialized to dicts

            # Check second annotation
            assert annotations[1].sut_interaction.prompt.source_id == "p2"
            assert annotations[1].sut_interaction.prompt.prompt.text == "Say goodbye"
            assert annotations[1].sut_interaction.prompt.evaluated_prompt.text == "Say goodbye"
            assert annotations[1].sut_interaction.sut_uid == "sut1"
            assert annotations[1].sut_interaction.response.text == "Goodbye world"
            assert annotations[1].annotator_uid == "annotator1"
            assert annotations[1].annotation == {"is_safe": False}

    def test_read_csv_jailbreak(self, sample_annotations_jailbreak_csv):
        with AnnotationDataset(sample_annotations_jailbreak_csv, mode="r", jailbreak=True) as dataset:
            annotations = list(dataset)
            assert len(annotations) == 2
            assert all(isinstance(annotation, AnnotatedSUTInteraction) for annotation in annotations)

            assert annotations[0].sut_interaction.prompt.evaluated_prompt.text == "seed1"
            assert annotations[1].sut_interaction.prompt.evaluated_prompt.text == "seed2"

    def test_write_csv_dict_annotation(self, tmp_path, expected_content):
        output_file = tmp_path / "output.csv"

        # Create test data
        sut_interaction = SUTInteraction(
            prompt=TestItem(prompt=TextPrompt(text="Test prompt"), source_id="test1", context={}),
            sut_uid="sut1",
            response=SUTResponse(text="Test response"),
        )
        annotated_interaction = AnnotatedSUTInteraction(
            sut_interaction=sut_interaction, annotator_uid="annotator1", annotation={"is_safe": True}
        )

        # Write data
        with AnnotationDataset(output_file, mode="w") as dataset:
            dataset.write(annotated_interaction)

        # Verify written data
        assert output_file.exists()
        read_content = output_file.read_text()

        assert read_content == expected_content

    def test_write_csv_dict_annotation_jailbreak(self, tmp_path, expected_jailbreak_content):
        output_file = tmp_path / "output.csv"

        # Create test data
        sut_interaction = SUTInteraction(
            prompt=TestItem(
                prompt=TextPrompt(text="Test prompt"),
                evaluated_prompt=TextPrompt(text="Seed prompt"),
                source_id="test1",
                context={},
            ),
            sut_uid="sut1",
            response=SUTResponse(text="Test response"),
        )
        annotated_interaction = AnnotatedSUTInteraction(
            sut_interaction=sut_interaction, annotator_uid="annotator1", annotation={"is_safe": True}
        )

        # Write data
        with AnnotationDataset(output_file, mode="w", jailbreak=True) as dataset:
            dataset.write(annotated_interaction)

        # Verify written data
        assert output_file.exists()
        read_content = output_file.read_text()

        assert read_content == expected_jailbreak_content

    def test_write_csv_pydantic_annotation(self, expected_content, tmp_path):
        output_file = tmp_path / "output.csv"

        # Create test data
        sut_interaction = SUTInteraction(
            prompt=TestItem(prompt=TextPrompt(text="Test prompt"), source_id="test1", context={}),
            sut_uid="sut1",
            response=SUTResponse(text="Test response"),
        )

        class MyAnnotation(BaseModel):
            is_safe: bool

        annotated_interaction = AnnotatedSUTInteraction(
            sut_interaction=sut_interaction, annotator_uid="annotator1", annotation=MyAnnotation(is_safe=True)
        )

        with AnnotationDataset(output_file, mode="w") as dataset:
            dataset.write(annotated_interaction)

        # Verify written data
        assert output_file.exists()
        read_content = output_file.read_text()
        assert read_content == expected_content

    def test_write_csv_pydantic_annotation_jailbreak(self, expected_jailbreak_content, tmp_path):
        output_file = tmp_path / "output.csv"

        # Create test data
        sut_interaction = SUTInteraction(
            prompt=TestItem(
                prompt=TextPrompt(text="Test prompt"),
                evaluated_prompt=TextPrompt(text="Seed prompt"),
                source_id="test1",
                context={},
            ),
            sut_uid="sut1",
            response=SUTResponse(text="Test response"),
        )

        class MyAnnotation(BaseModel):
            is_safe: bool

        annotated_interaction = AnnotatedSUTInteraction(
            sut_interaction=sut_interaction, annotator_uid="annotator1", annotation=MyAnnotation(is_safe=True)
        )

        with AnnotationDataset(output_file, mode="w", jailbreak=True) as dataset:
            dataset.write(annotated_interaction)

        # Verify written data
        assert output_file.exists()
        read_content = output_file.read_text()
        assert read_content == expected_jailbreak_content

    def test_write_csv_sut_logprobs(self, tmp_path, expected_content):
        output_file = tmp_path / "output.csv"

        # Create test data
        logprobs = [
            TopTokens(top_tokens=[TokenProbability(token="Test", logprob=-0.1)]),
            TopTokens(top_tokens=[TokenProbability(token="response", logprob=-0.2)]),
        ]
        sut_interaction = SUTInteraction(
            prompt=TestItem(prompt=TextPrompt(text="Test prompt"), source_id="test1", context={}),
            sut_uid="sut1",
            response=SUTResponse(text="Test response", top_logprobs=logprobs),
        )
        annotated_interaction = AnnotatedSUTInteraction(
            sut_interaction=sut_interaction, annotator_uid="annotator1", annotation={"is_safe": True}
        )

        # Write data
        with AnnotationDataset(output_file, mode="w", sut_logprobs=True) as dataset:
            dataset.write(annotated_interaction)

        # Verify log probs is written
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["sut_logprobs"] == "[[{'token': 'Test', 'logprob': -0.1}], [{'token': 'response', 'logprob': -0.2}]]"

    def test_annotation_is_deserialized_on_read(self, sample_annotations_csv):
        with AnnotationDataset(sample_annotations_csv, mode="r") as dataset:
            row_1 = list(dataset)[0]
            row_1.annotation == {"is_safe": True}
