import time
from collections import deque
from pathlib import Path

import cv2
import face_recognition
import streamlit as st
from image_utils import rescale_width_height

from models.face import Face
from utils.image_readers import open_image


def detect_faces(img_paths: list[Path], img_size: int) -> deque[Face]:
    """
    Detects faces and get face locations and encodings in images.
    :param img_paths: list of image paths.
    :param img_size: number of pixels to resize the longest edge to for inference.
    :return: instances of class Face in a queue (collections.deque()).
    """

    # initialize status bar
    _status_bar = st.progress(0, 'Firing up the face detection algorithm!')
    time.sleep(1)
    # keep a queue of found faces, the queue is a list of instances of class Face
    q = deque()
    # iterate over all image paths is the selected directory and gather all detected faces and face encodings
    for i, path in enumerate(img_paths):
        # update progress
        _status_bar.progress((i + 1) / len(img_paths),
                             text=f'({i + 1} of {len(img_paths)}) Detecting faces in {path.name}')
        # open image
        # image = cv2.imread(img_paths[i])
        image = open_image(image_path=path)
        # convert image color from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize the image for fast inference
        _w, _h = rescale_width_height(width=image.shape[1], height=image.shape[0], size=img_size)
        resized_image = cv2.resize(image, dsize=(_w, _h), interpolation=cv2.INTER_AREA)
        # detect face locations in the resized image
        face_locations = face_recognition.face_locations(resized_image, model='hog')
        for face_location in face_locations:
            # get face encoding
            encodings = face_recognition.face_encodings(resized_image,
                                                        known_face_locations=[face_location],
                                                        num_jitters=1,
                                                        model="large")
            # update the queue with a new instance of class Face
            q.append(
                Face(
                    img_path=path,
                    img_width=image.shape[1],
                    img_height=image.shape[0],
                    img_resized_width=resized_image.shape[1],
                    img_resized_height=resized_image.shape[0],
                    face_location=face_location,
                    encoding=encodings
                    )
                )

    _status_bar.empty()

    return q
