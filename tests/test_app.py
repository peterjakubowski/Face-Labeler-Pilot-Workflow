from streamlit.testing.v1 import AppTest


class TestAppStartUp:

    def test_smoke(self, at: AppTest):

        assert not at.exception, "App should run without any exceptions."
