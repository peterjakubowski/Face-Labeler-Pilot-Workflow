from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from config import COMPARE_FACES_TOLERANCE, TOP_K

SETTINGS_PAGE_PATH = Path("pages/settings.py")


class TestSettingsStartUp:

    def test_app_switch_pages_to_settings_page(self, at: AppTest):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert not at.exception, (
            "App should switch to settings page without any exceptions"
        )

    def test_app_loads_settings_page_elements(self, at: AppTest):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert at.title[0].value.startswith("Face Recognition Settings")
        assert at.markdown[0].value.startswith(
            "Control how the system recognizes faces."
        )
        assert len(at.number_input) == 2, (
            "App settings page should display two number inputs"
        )
        assert at.number_input[0].label == "Top K"
        assert at.number_input[1].label == "Threshold"
        assert len(at.info) == 1, "App settings page should display one info widget"
        assert at.info[0].value.startswith("Face")
        assert len(at.button) == 1, "App settings page should display one button"


class TestSettingsSessionState:

    def test_app_settings_page_session_state_updates_with_top_k_number_input(self, at: AppTest):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert 'top_k' not in at.session_state

        top_k_number_input = at.number_input[0]

        assert top_k_number_input.value == TOP_K

        top_k_number_input.increment().run()

        assert at.button[0].label.startswith("Save settings")

        at.button[0].click().run()

        assert 'top_k' in at.session_state
        assert at.session_state.top_k == TOP_K + 1

    def test_app_settings_page_session_state_updates_with_threshold_input(self, at: AppTest):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert 'threshold' not in at.session_state

        threshold_number_input = at.number_input[1]

        assert threshold_number_input.value == COMPARE_FACES_TOLERANCE

        threshold_number_input.increment().run()

        assert at.button[0].label.startswith("Save settings")

        at.button[0].click().run()

        expected_value = round(COMPARE_FACES_TOLERANCE + 0.05, 2)

        assert 'threshold' in at.session_state
        assert at.session_state.threshold == expected_value


class TestLoadLibrary:

    def test_app_settings_page_loads_with_the_face_classifier_not_initialized(self, at: AppTest):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert len(at.info) == 1
        assert at.info[0].value == "Face classifier is uninitialized."

    def test_app_settings_loads_library_when_directory_is_empty(self, at: AppTest, mock_library_dir_path: Path):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert len(at.button) == 1
        assert at.button[0].label == "Load Faces Library"

        at.button[0].click().run()

        assert at.info[0].value == "Face classifier contains **0** total face embeddings and **0** unique names."

    def test_app_settings_reset_faces_library_button(self, at: AppTest, mock_library_dir_path: Path):

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert len(at.button) == 1
        assert at.button[0].label == "Load Faces Library"

        at.button[0].click().run()

        # NOTE: There should be a st.popover button here, but it is not testable with AppTest

        assert len(at.button) == 1
        assert at.button[0].label == "Confirm Reset"

        at.button[0].click().run()

        assert at.info[0].value == "Face classifier is uninitialized."

    def test_app_settings_loads_library_when_directory_has_one_image_with_no_face(self, at: AppTest, mock_library_dir_path: Path, monkeypatch):

        new_image_name = mock_library_dir_path / "test_image_1.jpg"
        new_image = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(new_image_name), new_image)

        monkeypatch.setattr(
            "utils.library.extract_metadata_from_files_with_exiftool",
            lambda *args: []
        )

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert at.button[0].label == "Load Faces Library"

        at.button[0].click().run()

        assert at.info[0].value == "Face classifier contains **0** total face embeddings and **0** unique names."

    def test_app_settings_loads_library_when_directory_has_one_image_with_one_face(self, at: AppTest, mock_library_dir_path: Path, monkeypatch):
        # create a new blank image and save it in the temp library folder
        new_image_name = mock_library_dir_path / "test_image_1.jpg"
        new_image = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(new_image_name), new_image)
        # mock extracted metadata with a face region
        mock_face_metadata = {
            'SourceFile': new_image_name,
            'XMP:RegionType': 'Face',
            'XMP:RegionName': 'Person Shown Name',
            'XMP:RegionAreaW': 0.0,
            'XMP:RegionAreaH': 0.0,
            'XMP:RegionAreaX': 0.0,
            'XMP:RegionAreaY': 0.0
        }

        monkeypatch.setattr(
            "utils.library.extract_metadata_from_files_with_exiftool",
            lambda *args: [mock_face_metadata]
        )
        # mock a face recognition face encoding
        # NOTE: while the encoding is mocked here, it's important to note that the `face_encodings` method
        # returns an encoding no matter what when a face location is provided, so mocking this isn't
        # completely necessary. Since we've created an empty image, and there is no face in it, I would expect
        # the method to have trouble generating an encoding, yet it seems to return an encoding regardless
        # what's in the image.
        mock_face_encoding = np.random.rand(128)

        monkeypatch.setattr(
            "utils.library.face_recognition.face_encodings",
            lambda *args, **kwargs: [mock_face_encoding]
        )

        at.switch_page(str(SETTINGS_PAGE_PATH)).run()

        assert at.button[0].label == "Load Faces Library"

        at.button[0].click().run()

        assert at.info[0].value == "Face classifier contains **1** total face embeddings and **1** unique names."
