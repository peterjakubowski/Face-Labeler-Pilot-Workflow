from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from utils.face_classifier import face_conn

APP_FILE_PATH = Path("app.py")


@pytest.fixture
def mock_img_dir_path(monkeypatch, tmp_path) -> Path:

    temp_watch_folder_path = tmp_path / "watch_folder"

    temp_watch_folder_path.mkdir(exist_ok=True)

    monkeypatch.setattr("utils.helpers.IMG_DIR", temp_watch_folder_path)

    return temp_watch_folder_path


@pytest.fixture
def mock_library_dir_path(monkeypatch, tmp_path) -> Path:

    temp_library_path = tmp_path / "library"

    temp_library_path.mkdir(exist_ok=True)

    monkeypatch.setattr("utils.library.LIB_DIR", temp_library_path)

    return temp_library_path


@pytest.fixture
def mock_face_recognition_face_location(monkeypatch):

    top, right, bottom, left = (0, 10, 10, 0)

    mock_face_locations = [(top, right, bottom, left)]

    monkeypatch.setattr(
        "utils.image_processing.face_recognition.face_locations",
        lambda *args, **kwargs: mock_face_locations)


@pytest.fixture
def mock_face_recognition_encodings(monkeypatch):

    mock_face_encoding = np.random.rand(128)

    monkeypatch.setattr(
        "utils.image_processing.face_recognition.face_encodings",
        lambda *args, **kwargs: [mock_face_encoding])


@pytest.fixture
def at(request):

    app_test = AppTest.from_file(APP_FILE_PATH).run()

    # if "manual_run" not in request.keywords:
    #     app_test.run()

    yield app_test

    # ensure our cache and connection is cleared
    st.cache_data.clear()
    st.cache_resource.clear()
    face_conn.reset()
