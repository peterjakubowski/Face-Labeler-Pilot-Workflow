from pathlib import Path

from streamlit.testing.v1 import AppTest

VIEWER_PAGE_PATH = Path("pages/viewer.py")


class TestViewerStartUp:

    def test_app_switch_pages_to_viewer_page(self, at: AppTest):

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert not at.exception, "App should switch to viewer pages without any exceptions"

    def test_app_loads_viewer_page_elements(self, at: AppTest):

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert at.title[0].value.startswith("Image Viewer")
        assert at.markdown[0].value.startswith("Extracts embedded metadata from images and displays labeled faces")
        assert at.selectbox[0].label.startswith("Choose a folder")
        assert at.button[0].label.startswith("View Images")
