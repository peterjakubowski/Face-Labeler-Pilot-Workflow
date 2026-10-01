from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

VIEWER_PAGE_PATH = Path("pages/viewer.py")


class TestViewerStartUp:
    def test_app_switch_pages_to_viewer_page(self, at: AppTest):

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert not at.exception, (
            "App should switch to viewer pages without any exceptions"
        )

    def test_app_loads_viewer_page_elements(self, at: AppTest):

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert at.title[0].value.startswith("Image Viewer")
        assert at.markdown[0].value.startswith(
            "Extracts embedded metadata from images and displays labeled faces"
        )
        assert at.selectbox[0].label.startswith("Choose a folder")
        assert at.button[0].label.startswith("View Images")


class TestViewerDisplaysImages:

    def test_app_viewer_shows_folder_name_select_box_with_no_options(self, at:AppTest, mock_img_dir_path: Path):

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert len(at.selectbox) == 1
        assert at.selectbox[0].label.startswith("Choose a folder of images")
        assert at.selectbox[0].options == []
        assert len(at.warning) == 1, "App should display warning message"
        assert at.warning[0].value.startswith("The watch folder is empty")

    def test_app_viewer_page_displays_one_image(self, at: AppTest, mock_img_dir_path: Path, monkeypatch):

        at.run()

        test_folder_1 = mock_img_dir_path / "test folder 1"
        test_folder_1.mkdir()

        new_image_name = test_folder_1 / "test_image_1.jpg"
        new_image = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(new_image_name), new_image)

        mock_extracted_metadata = {
            'SourceFile': new_image_name
        }

        monkeypatch.setattr(
            "utils.helpers.extract_metadata_from_files_with_exiftool",
            lambda *args: [mock_extracted_metadata]
        )

        at.switch_page(str(VIEWER_PAGE_PATH)).run()

        assert len(at.selectbox[0].options) == 1
        assert at.selectbox[0].options[0] == "test folder 1"
        assert len(at.button) == 1, "App viewer should display one button"
        assert at.button[0].label == "View Images"

        at.selectbox[0].set_value("test folder 1").run()
        at.button[0].click().run()

        assert len(at.image) == 1, "App viewer should display one image"