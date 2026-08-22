from models.face import Face
import cv2
import streamlit as st
import face_recognition
import time
from collections import deque


def rescale_width_height(width: int, height: int, size: int) -> tuple[int, ...]:
    """
    Function for rescaling the width and height
    of an image to keep aspect ratio.
    :param width: original image width
    :param height: original image height
    :param size: desired length of the longest edge in pixels.
    :return: width (w) and height (h) of resized image.
    """

    # check if the image is vertical,
    # height is the longest edge
    if height > width:
        # set height to size
        h = size
        # determine the ratio for resizing
        ratio = height / size
        # calculate new width by dividing by ratio
        w = int(width / ratio)
    # check if the image is horizontal,
    # width is the longest edge
    elif height < width:
        # set width to size
        w = size
        # determine the ratio for resizing
        ratio = width / size
        # calculate new height by dividing by ratio
        h = int(height / ratio)
    # if image is not vertical or horizontal,
    # image must be square
    else:
        # set width and height to size
        w = h = size
    # return the new width and height
    return tuple([w, h])


def detect_faces(img_paths: list, img_size: int) -> deque:
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
    for i in range(len(img_paths)):
        # update progress
        _status_bar.progress((i + 1) / len(img_paths),
                             text=f'({i + 1} of {len(img_paths)}) Detecting faces in {img_paths[i].split("/")[-1]}...')
        # open image
        image = cv2.imread(img_paths[i])
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
            q.append(Face(img_path=img_paths[i],
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
