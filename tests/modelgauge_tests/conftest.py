import logging

import pytest
from modelgauge_tests.utilities import parent_directory

from modelgauge import command_line


@pytest.fixture
def handled_caplog(caplog, monkeypatch):

    original_configure = command_line.configure_logging

    def patched_configure_logging(*args, **kwargs):
        original_configure(*args, **kwargs)
        root_logger = logging.getLogger()
        if caplog.handler not in root_logger.handlers:
            formatter = root_logger.handlers[0].formatter
            caplog.handler.setFormatter(formatter)
            root_logger.addHandler(caplog.handler)

    monkeypatch.setattr(command_line, "configure_logging", patched_configure_logging)
    return caplog
