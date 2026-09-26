from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_FILE_PATH = Path("app.py")
SETTINGS_PAGE_PATH = Path("pages/settings.py")
VIEWER_PAGE_PATH = Path("pages/viewer.py")


@pytest.fixture
def at():

    app_test = AppTest.from_file(APP_FILE_PATH)
    app_test.run()
    return app_test


class TestStartUp:

    def test_smoke(self, at: AppTest):

        assert not at.exception, "App should run without any exceptions."

    def test_app_launches_to_workflow_page_with_title_and_markdown(self, at: AppTest):

        assert at.title[0].value.startswith("Face Labeler Pilot")
        assert at.markdown[0].value.startswith("Face Labeler Pilot is a 3-step post-production workflow tool")
        assert at.subheader[0].value.startswith("Step 1: Detect Faces")
        assert at.selectbox[0].label.startswith("Choose a folder of images")

    def test_app_switch_pages_to_settings_page(self, at: AppTest):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert not at.exception, "App should switch to settings page without any exceptions"

    def test_app_switch_pages_to_viewer_page(self, at: AppTest):

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert not at.exception, "App should switch to viewer pages without any exceptions"

    def test_app_loads_settings_page_elements(self, at: AppTest):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert at.title[0].value.startswith("Face Recognition Settings")
        assert at.markdown[0].value.startswith("Control how the system recognizes faces.")
        assert len(at.number_input) == 2, "App settings page should display two number inputs"
        assert at.number_input[0].label == "Top K"
        assert at.number_input[1].label == "Threshold"
        assert len(at.info) == 1, "App settings page should display one info widget"
        assert at.info[0].value.startswith("Face")
        assert len(at.button) == 1, "App settings page should display one button"

    def test_app_loads_viewer_page_elements(self, at: AppTest):

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert at.title[0].value.startswith("Image Viewer")
        assert at.markdown[0].value.startswith("Extracts embedded metadata from images and displays labeled faces")
        assert at.selectbox[0].label.startswith("Choose a folder")
        assert at.button[0].label.startswith("View Images")
