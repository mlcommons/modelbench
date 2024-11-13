import json
import pytest
from typing import Dict, List

from modelbench.consistency_checker import ConsistencyChecker


def make_basic_run(suts: List[str], test_prompts: Dict[str, List[str]], annotators: List[str]):
    """Successful "fresh" benchmark run with all SUT/annotator responses fetched (not cached)."""
    journal = []
    journal.append({"message": "starting run", "suts": suts, "tests": list(test_prompts.keys())})
    for sut in suts:
        for test, prompts in test_prompts.items():
            journal.append({"message": "using test items", "test": test, "using": len(prompts)})
            for prompt in prompts:
                # Normal pipeline.
                sut_messages = [
                    "queuing item",
                    "fetched sut response",
                    "translated sut response",
                    "measured item quality",
                ]
                for message in sut_messages:
                    journal.append({"message": message, "test": test, "prompt_id": prompt, "sut": sut})
                for annotator in annotators:
                    journal.append(
                        {
                            "message": "fetched annotator response",
                            "test": test,
                            "sut": sut,
                            "prompt_id": prompt,
                            "annotator": annotator,
                        }
                    )
    return journal


@pytest.fixture
def basic_benchmark_run():
    return make_basic_run(
        suts=["sut1", "sut2"], test_prompts={"test1": ["prompt1", "prompt2"]}, annotators=["annotator1", "annotator2"]
    )


def test_normal_run(tmp_path, basic_benchmark_run):
    journal_path = tmp_path / "journal.jsonl"
    with open(journal_path, "w") as f:
        for item in basic_benchmark_run:
            f.write(json.dumps(item) + "\n")

    checker = ConsistencyChecker(journal_path=journal_path)
    checker.run()

    for subchecker in [checker.test_sut_level_checker, checker.test_sut_annotator_level_checker]:
        assert subchecker.warnings == []
        assert subchecker.check_is_complete()
