from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


class TestWorkflowStartUp:
    def test_app_launches_to_workflow_page_with_title_and_markdown(self, at: AppTest):

        assert at.title[0].value.startswith("Face Labeler Pilot")
        assert at.markdown[0].value.startswith(
            "Face Labeler Pilot is a 3-step post-production workflow tool"
        )
        assert at.subheader[0].value.startswith("Step 1: Detect Faces")
        assert at.selectbox[0].label.startswith("Choose a folder of images")
        assert at.selectbox[0].value is None
        assert len(at.button) == 0, "App should not display button at load"


class TestWorkflowSessionState:
    def test_app_workflow_page_session_state_is_initially_empty(self, at: AppTest):

        # set when selecting a folder from the select box
        assert "select_folder" not in at.session_state
        # set during run face detection workflow
        assert "image_paths" not in at.session_state
        assert "faces_detected" not in at.session_state
        assert "faces_count" not in at.session_state
        assert "face_i" not in at.session_state
        assert "labeled" not in at.session_state
        assert "name_options" not in at.session_state
        # set on the settings page
        assert "top_k" not in at.session_state
        assert "threshold" not in at.session_state

    def test_app_workflow_page_session_state_initializes_after_detecting_faces(
        self,
        at: AppTest,
        mock_img_dir_path: Path,
        mock_face_recognition_face_location,
        mock_face_recognition_encodings,
    ):

        test_folder_1 = mock_img_dir_path / "test folder 1"
        test_folder_1.mkdir()

        new_image_name = test_folder_1 / "test_image_1.jpg"
        new_image = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(new_image_name), new_image)

        at.run()

        at.selectbox[0].select("test folder 1").run()
        at.button[0].click().run()

        assert "select_folder" in at.session_state
        assert at.session_state.select_folder == "test folder 1"

        assert "image_paths" in at.session_state
        assert "faces_detected" in at.session_state
        assert "faces_count" in at.session_state
        assert "face_i" in at.session_state
        assert "labeled" in at.session_state
        assert "name_options" in at.session_state


class TestWorkflowSelectBox:

    def test_app_workflow_shows_folder_name_select_box_with_no_options(
        self, at: AppTest, mock_img_dir_path: Path
    ):

        at.run()

        assert len(at.selectbox) == 1
        assert at.selectbox[0].label.startswith("Choose a folder of images")
        assert at.selectbox[0].options == []
        assert len(at.warning) == 1, "App should display warning message"
        assert at.warning[0].value.startswith("The watch folder is empty")

    def test_app_workflow_shows_folder_name_select_box_with_options(
        self, at: AppTest, mock_img_dir_path: Path
    ):

        test_folder_1 = mock_img_dir_path / "test folder 1"

        test_folder_1.mkdir()

        at.run()

        assert len(at.selectbox[0].options) == 1
        assert at.selectbox[0].options[0] == "test folder 1"
        assert len(at.warning) == 0, "App should not display warning message"


class TestRunFaceDetectionWorkflow:

    def test_app_workflow_runs_step_1_face_detection_finds_no_faces_empty_folder(
        self, at: AppTest, mock_img_dir_path: Path
    ):

        test_folder_1 = mock_img_dir_path / "test folder 1"
        test_folder_1.mkdir()

        at.run()

        at.selectbox[0].select("test folder 1").run()
        at.button[0].click().run()

        assert (
            at.warning[0].value
            == "Workflow complete! Looked for faces in 0 images and 0 faces were found."
        )

    def test_app_workflow_runs_step_1_face_detection_finds_no_faces_from_one_image(
        self, at: AppTest, mock_img_dir_path: Path
    ):

        test_folder_1 = mock_img_dir_path / "test folder 1"
        test_folder_1.mkdir()

        new_image_name = test_folder_1 / "test_image_1.jpg"
        new_image = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(new_image_name), new_image)

        at.run()

        at.selectbox[0].select("test folder 1").run()
        at.button[0].click().run()

        assert (
            at.warning[0].value
            == "Workflow complete! Looked for faces in 1 images and 0 faces were found."
        )

    def test_app_workflow_runs_step_1_face_detection_finds_one_face_from_one_image(
        self,
        at: AppTest,
        mock_img_dir_path: Path,
        mock_face_recognition_face_location,
        mock_face_recognition_encodings,
    ):

        test_folder_1 = mock_img_dir_path / "test folder 1"
        test_folder_1.mkdir()

        new_image_name = test_folder_1 / "test_image_1.jpg"
        new_image = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(new_image_name), new_image)

        at.run()

        at.selectbox[0].select("test folder 1").run()
        at.button[0].click().run()

        assert (
            at.success[0].value
            == "Face detection is complete! Found 1 faces in 1 images."
        )

    def test_app_workflow_runs_step_2_face_recognition_one_unrecognized_face_from_one_image(
        self,
        at: AppTest,
        mock_img_dir_path: Path,
        mock_face_recognition_face_location,
        mock_face_recognition_encodings,
    ):

        test_folder_1 = mock_img_dir_path / "test folder 1"
        test_folder_1.mkdir()

        new_image_name = test_folder_1 / "test_image_1.jpg"
        new_image = np.zeros((100, 100, 3), dtype=np.uint8)

        cv2.imwrite(str(new_image_name), new_image)

        at.run()

        at.selectbox[0].select("test folder 1").run()
        at.button[0].click().run()

        assert len(at.subheader) == 2
        assert at.subheader[1].value == "Step 2: Label Faces"

        assert len(at.checkbox) == 1, "App should display one check box"
        assert at.checkbox[0].label == "Auto confirm matches?"
        assert at.checkbox[0].value is False

        # assert len(at.progress) == 1, "App should display progress bar"
        # assert at.progress[0].label == "Labeling face 1 of 1"

        assert len(at.image) == 1, "App should display one image face thumbnail"

        assert len(at.markdown) == 2
        assert at.markdown[1].value.startswith("I don't recognize this face")

        assert len(at.selectbox) == 2
        assert at.selectbox[1].label.startswith(
            "Type in a new name or select one from the list"
        )
        assert at.selectbox[1].value is None

        assert len(at.button) == 2
        assert at.button[1].label == "Submit"

