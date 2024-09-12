import csv
import io
import os
from modelgauge.dependency_helper import DependencyHelper
from typing import List, Mapping


class FakeDependencyHelper(DependencyHelper):
    """Test version of Dependency helper that lets you set the text in files.

    If the "value" in dependencies is a string, this will create a file with "value" contents.
    If the "value" is a Mapping, it will treat those as file name + content pairs.
    """

    def __init__(self, tmpdir, dependencies: Mapping[str, str | Mapping[str, str]]):
        self.tmpdir = tmpdir
        # Create each of the files.
        for key, dependency in dependencies.items():
            if isinstance(dependency, str):
                with open(os.path.join(tmpdir, key), "w") as f:
                    f.write(dependency)
            else:
                for subfile_name, subfile_contents in dependency.items():
                    with open(os.path.join(tmpdir, key, subfile_name), "w") as f:
                        f.write(subfile_contents)
        self.dependencies = dependencies

    def get_local_path(self, dependency_key: str) -> str:
        assert dependency_key in self.dependencies, (
            f"Key {dependency_key} is not one of the known "
            f"dependencies: {list(self.dependencies.keys())}."
        )
        return os.path.join(self.tmpdir, dependency_key)

    def versions_used(self) -> Mapping[str, str]:
        raise NotImplementedError("Fake isn't implemented for this yet.")

    def update_all_dependencies(self) -> Mapping[str, str]:
        raise NotImplementedError("Fake isn't implemented for this yet.")


def make_csv(header: List[str], rows: List[List[str]]) -> str:
    """Construct csv valid text from the header and rows."""
    # Check that data is set up as expected
    for row in rows:
        assert len(row) == len(header)
    # Handles quoting and escaping of delimiters
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows([header, *rows])
    return output.getvalue()
