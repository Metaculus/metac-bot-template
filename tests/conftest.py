import pytest
from unittest.mock import patch

@pytest.fixture
def mock_os_getenv():
    with patch('os.getenv') as mock_getenv:
        yield mock_getenv
