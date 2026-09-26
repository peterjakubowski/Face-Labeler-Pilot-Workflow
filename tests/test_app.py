from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_FILE_PATH = Path("app.py")


@pytest.fixture
def at():

    app_test = AppTest.from_file(APP_FILE_PATH)
    app_test.run()
    return app_test


class TestStartUp:

    def test_smoke(self, at: AppTest):

        assert not at.exception, "App should run without any exceptions."
