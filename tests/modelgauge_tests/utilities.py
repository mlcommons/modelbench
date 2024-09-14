import pathlib
import pytest

expensive_tests = pytest.mark.skipif("not config.getoption('expensive-tests')")


@pytest.fixture
def parent_directory(request):
    """Pytest fixture that returns the parent directory of the currently executing test file."""
    file = pathlib.Path(request.node.fspath)
    return file.parent
