import time
from collections import deque
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import streamlit as st
from image_utils import rescale_width_height

from config import IMG_PREVIEW_WIDTH, IMG_SIZE
from models.face import Face
from utils.image_readers import open_image


def prepare_image_for_inference(image_path: Path, img_size: int = IMG_SIZE) -> tuple[tuple[int, ...], np.ndarray]:
    """
    Opens an image from a given path and returns a numpy array ready for inference.

    :param image_path: Path to image
    :param img_size: Number of pixels to resize the longest edge to for inference.
    :return: Image as numpy array
    """

    # open image
    image = open_image(image_path=image_path)
    # convert image color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize the image for fast inference
    _w, _h = rescale_width_height(
        width=image.shape[1], height=image.shape[0], size=img_size
    )
    resized_image = cv2.resize(image, dsize=(_w, _h), interpolation=cv2.INTER_AREA)

    return image.shape, resized_image


def prepare_image_for_annotation(image_path: Path) -> np.ndarray:

    # img = cv2.imread(m["SourceFile"])
    img = open_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    height = int((img_height / img_width) * IMG_PREVIEW_WIDTH)
    img = cv2.resize(img, dsize=(IMG_PREVIEW_WIDTH, height), interpolation=cv2.INTER_AREA)

    return img


def annotate_image_with_face_region_using_opencv(img: np.ndarray, person_shown: str, w: int, h: int, x: int, y: int) -> np.ndarray:
    """
    Annotate an image by surrounding a face region with a bounding box and label.

    :param img: Image as numpy array
    :param person_shown: Person (face) shown in the image region
    :param w: XMP Region Area W
    :param h: XMP Region Area H
    :param x: XMP Region Area X
    :param y: XMP Region Area Y
    :return: Image as numpy array
    """

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # get text size
    text_size = cv2.getTextSize(person_shown, cv2.FONT_HERSHEY_PLAIN, 1.3, 2)
    dim = text_size[0]
    baseline = text_size[1]
    # Use text size to create a black rectangle
    cv2.rectangle(
        img,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )
    # put text labels on the image
    cv2.putText(
        img, person_shown, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2
    )

    return img


def detect_faces(img_paths: list[Path]) -> deque[Face]:
    """
    Detects faces and get face locations and encodings in images.
    :param img_paths: list of image paths.
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
        shape, image = prepare_image_for_inference(image_path=path)
        # detect face locations in the image
        face_locations = face_recognition.face_locations(image, model='hog')
        # iterate over all detected faces
        for face_location in face_locations:
            # get face encoding
            encodings = face_recognition.face_encodings(image,
                                                        known_face_locations=[face_location],
                                                        num_jitters=1,
                                                        model="large")
            # update the queue with a new instance of class Face
            q.append(
                Face(
                    img_path=path,
                    img_width=shape[1],
                    img_height=shape[0],
                    img_resized_width=image.shape[1],
                    img_resized_height=image.shape[0],
                    face_location=face_location,
                    encoding=encodings
                    )
                )

    _status_bar.empty()

    return q
