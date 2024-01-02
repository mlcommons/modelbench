import pathlib
import pytest


@pytest.fixture
def parent_directory(request):
    """Pytest fixture that returns the parent directory of the currently executing test file."""
    file = pathlib.Path(request.node.fspath)
    return file.parent
