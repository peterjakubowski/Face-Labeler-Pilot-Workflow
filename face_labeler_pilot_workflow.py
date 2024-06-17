# Face Labeler Pilot is a Python-based photography workflow tool
# for tagging people shown in images using face recognition.
#
# Author: Peter Jakubowski
# Date: 5/9/2024
# Description: Streamlit app that opens a selected folder of images
# and detects faces for labeling.
#
#
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import session_state as sess
from streamlit_free_text_select import st_free_text_select
from imutils import paths
import face_recognition
import cv2
import exiftool
import os
from pathlib import Path
import uuid
import time
from collections import defaultdict
from collections import deque
import csv


class Face:
    """
    structure to store information about detected faces
    """

    def __init__(self,
                 img_path: str,
                 img_width: int,
                 img_height: int,
                 img_resized_width: int,
                 img_resized_height: int,
                 face_location: tuple[int, ...],
                 encoding: list) -> None:
        """
        Constructor for Face class objects.
        :param img_path: path to the image the face was detected in.
        :param img_width: original image width in pixels.
        :param img_height: original image height in pixels.
        :param img_resized_width: resized image width in pixels.
        :param img_resized_height: resized image height in pixels.
        :param face_location: location of the face in the image (top, right, bottom, left).
        :param encoding: encoding of the face detected in the image.
        """

        self.img_path = img_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_resized_width = img_resized_width
        self.img_resized_height = img_resized_height
        self.face_location = face_location
        self.encoding = encoding
        self.match_candidate = True
        self.person_shown = ""
        self.W = None
        self.H = None
        self.X = None
        self.Y = None
        self.normalize_face_location()

    def open_face_image(self) -> np.ndarray:
        """
        Opens the image containing the current face using cv2
        and crops the image to the region the face is in.
        :return: image (numpy.ndarray) cropped to the current face.
        """

        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _W, _H, _X, _Y = self.reverse_transform_face_location(width=img.shape[1], height=img.shape[0])
        img = img[_Y:_Y + _H, _X:_X + _W]
        return img

    def normalize_face_location(self) -> None:
        """
        Normalize/scale the face location coordinates.
        Bounding box (W, H, X, Y):
        Width of the bounding box.
        Height of the bounding box.
        X coordinate of the left of the bounding box.
        Y coordinate of the top of the bounding box.
        """

        top, right, bottom, left = self.face_location
        img_w, img_h = self.img_resized_width, self.img_resized_height
        self.W = round((right - left) / img_h, 4)
        self.H = round((bottom - top) / img_w, 4)
        self.X = round(left / img_h, 4)
        self.Y = round(top / img_w, 4)
        return

    def reverse_transform_face_location(self, width: int, height: int) -> tuple[int, ...]:
        """
        Reverse transform the normalized/scaled bounding box
        given the width and height of an image.
        :param width: width of image to scale bounding box to
        :param height: height of image to scale bounding box to
        :return: Reverse transformed bounding box (W, H, X, Y):
        Width of the bounding box.
        Height of the bounding box.
        X coordinate of the left of the bounding box.
        Y coordinate of the top of the bounding box.
        """

        if None in [self.W, self.H, self.X, self.Y]:
            self.normalize_face_location()
        _W = int(self.W * height)
        _H = int(self.H * width)
        _X = int(self.X * height)
        _Y = int(self.Y * width)
        return tuple([_W, _H, _X, _Y])


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


def record_name() -> None:
    if sess.selected_name:
        _current_face = sess['faces_detected'][0]
        # if the user selected 'Someone else', that means the face recognition algorithm
        # predicted the name belonging to the face, but the user disagrees with the prediction
        # and wants to correct the name with a name that is not in the current list of names.
        # We mark the face's match candidate attribute False, so we can revisit this face and
        # enter a new name to the list. In doing this we provide the user with a new select widget (free_text_select).
        if sess.selected_name == 'Someone else':
            _current_face.match_candidate = False
        else:
            if sess.selected_name == 'Not a face':
                sess.faces_count -= 1
            else:
                _current_face.person_shown = sess.selected_name
                sess.name_options[_current_face.person_shown] += 1
                sess.face_i += 1
                sess.labeled[_current_face.img_path].append(_current_face)
                if len(_current_face.encoding) > 0:
                    sess.data['encodings'].append(_current_face.encoding[0])
                    sess.data['names'].append(_current_face.person_shown)
            sess['faces_detected'].popleft()
    return


# INFO: ===== Face Labeler Pilot Introduction ====
st.title("Face Labeler Pilot")
intro_text = ("Face Labeler Pilot is a 3-step post-production workflow tool "
              "that uses face recognition to tag people shown in photographs.")
st.markdown(intro_text)
# INFO: ===== Begin Step 1: Detect Faces ====
st.subheader("Step 1: Detect Faces", divider="gray")

# Set the image size for inference, the number of pixels the longest edge should be resized to
IMG_SIZE = 1024
# Set the path to the 'watch_folder' directory
IMG_DIR = Path("watch_folder")
# Make the 'watch_folder' directory if it does not exist
Path.mkdir(IMG_DIR, exist_ok=True)
# List the subfolders of the 'watch_folder'
folder_names = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
# Display a warning if there are no subfolders in the 'watch_folder'
if not folder_names:
    st.warning("The watch folder is empty. Add a folder of images to the watch folder to begin.")
# Streamlit select widget, gives the user a way to select a folder of images
select_folder = st.selectbox(label='Choose a folder of images to scan for faces',
                             index=None,
                             options=folder_names,
                             placeholder='Choose a folder of images',
                             label_visibility='collapsed'
                             )

if select_folder:
    # Streamlit button widget, kicks off the face detection workflow when pressed
    start_face_detection = st.button(label="Detect Faces")
    if start_face_detection:
        # list all the images (paths) in the selected folder
        sess['image_paths'] = sorted(paths.list_images(os.path.join(IMG_DIR, select_folder)),
                                     key=lambda x:x.split('/')[-1])
        # detect faces in all the images, get a list/queue of faces (instances of Face class)
        sess['faces_detected'] = detect_faces(img_paths=sess.image_paths, img_size=IMG_SIZE)
        # count how many faces were detected
        sess['faces_count'] = len(sess.faces_detected)
        # if we didn't detect any faces, delete the queue from the session state and display a message
        if sess.faces_count < 1:
            del sess['faces_detected']
            st.warning(f"Workflow complete! "
                       f"Looked for faces in {len(sess['image_paths'])} images "
                       f"and {sess['faces_count']} faces were found.")
        # count of faces labeled
        sess['face_i'] = 1
        # dictionary of labeled faces
        sess['labeled'] = defaultdict(list)
        # dictionary of face encodings and names
        sess['data'] = {'encodings': [], 'names': []}
        # dictionary of names/identities and counts
        sess['name_options'] = defaultdict(int)

if 'faces_detected' in sess:
    success_text = (f"Face detection is complete! "
                    f"Found {sess.faces_count} faces "
                    f"in {len(sess.image_paths)} images."
                    )
    st.success(success_text, icon='✅')

    # INFO: ===== Begin Step 2: Label Faces ====
    st.subheader("Step 2: Label Faces", divider="gray")

    # check if there are faces in our queue
    if sess['faces_detected']:
        # ask the user if the workflow should automatically confirm/accept matches
        auto_confirm_matches = st.checkbox(label="Auto confirm matches?",
                                           value=False,
                                           key='auto_confirm_matches')
        status_bar = st.progress(sess.face_i / sess.faces_count,
                                 text=f'Labeling face {sess.face_i} of {sess.faces_count}')
        # pop the next face from the queue
        current_face = sess['faces_detected'][0]
        # open cropped image of current face
        current_face_img = current_face.open_face_image()
        # check if the current face has an encoding
        if len(current_face.encoding) > 0:
            # compare the face encoding to existing encodings to see if we can find a match
            # note: the lower the tolerance, the more sensitive the algorithm is at matching faces
            matches = face_recognition.compare_faces(sess.data['encodings'],
                                                     current_face.encoding[0],
                                                     tolerance=0.55)

            if True in matches:
                matched_indices = [i for (i, b) in enumerate(matches) if b]
                count = {}
                for i in matched_indices:
                    name = sess.data['names'][i]
                    count[name] = count.get(name, 0) + 1
                predicted_name = max(count, key=count.get)
                if not auto_confirm_matches:
                    with st.form(str(uuid.uuid4())):
                        st.image(current_face_img, width=100)
                        if current_face.match_candidate:
                            st.write(f'I think this face belongs to **{predicted_name}**, can you confirm?')
                            st.selectbox(label=('The predicted name has been pre-selected, '
                                                'click the submit button to confirm.\n\n'
                                                'Select "Not a face" to skip this face.\n\n'
                                                'Select "Someone else" if their name '
                                                'is not in the list.\n'),
                                         options=['Not a face', 'Someone else'] + sorted(sess.name_options.keys()),
                                         index=sorted(sess.name_options.keys()).index(predicted_name) + 2,
                                         key='selected_name')
                        elif not current_face.match_candidate:
                            st.write((f"I think this face belongs to **{predicted_name}**, "
                                      "but you think it's someone else, who do you think this is?"))
                            st_free_text_select(label=('Who do you think this is? Type in a new name '
                                                       'or select one from the list. '
                                                       'Select "Not a face" to skip this face.'),
                                                options=sorted(sess.name_options.keys()),
                                                key="selected_name")
                        st.form_submit_button(label='Submit', on_click=record_name)
                elif auto_confirm_matches:
                    st.image(current_face_img, width=100)
                    st.write(f"This face belongs to **{predicted_name}**")
                    sess['selected_name'] = predicted_name
                    # wait for a moment, user can still interrupt by unchecking auto confirm matches
                    time.sleep(1)
                    record_name()
                    st.rerun()

            else:
                with st.form(str(uuid.uuid4())):
                    st.image(current_face_img, width=100)
                    st.write("I don't recognize this face, who is this?")
                    st_free_text_select(label=('Type in a new name or select one from the list. '
                                               'Select "Not a face" to skip this face.'),
                                        options=['Not a face'] + sorted(sess.name_options.keys()),
                                        key="selected_name")
                    st.form_submit_button(label='Submit', on_click=record_name)
        elif not current_face.encoding:
            with st.form(str(uuid.uuid4())):
                st.image(current_face_img, width=100)
                st.write('This face has no encoding. Is this a face?')
                st_free_text_select(label=('Type in a new name or select one from the list. '
                                           'Select "Not a face" to skip this face.'),
                                    options=['Not a face'] + sorted(sess.name_options.keys()),
                                    key="selected_name")
                st.form_submit_button(label='Continue', on_click=record_name)
    # if our queue of faces is empty, check if we have labeled any images
    if not sess['faces_detected']:
        # if we have labeled data, let's embed the face locations and names in the image metadata
        if 'labeled' in sess:
            if not sess['labeled']:
                st.success(f'{len(sess.labeled)} faces were labeled. Workflow complete!',
                           icon='✅')
            elif sess['labeled']:
                success_text = "All faces have been labeled!"
                st.success(success_text, icon='✅')
                df = pd.DataFrame(data=sess.name_options.items(),
                                  columns=['names', 'counts'])
                df.set_index('names', inplace=True)
                st.dataframe(df.sort_index())

                # INFO: ===== Begin Step 3: Write/Save/Embed Metadata ====
                st.subheader("Step 3: Save Metadata", divider="gray")
                col1, col2, _, _ = st.columns(4)
                with col1:
                    write_metadata = st.button(label="Write Metadata")
                with col2:
                    export_metadata = st.button(label="Export Metadata")

                if write_metadata:
                    status_text = 'Begin writing metadata to files!'
                    status_bar = st.progress(0, status_text)
                    time.sleep(1)
                    n = len(sess.labeled)
                    j = 0
                    for image_path, faces in sess.labeled.items():
                        status_bar.progress((j + 1) / n,
                                            text=f'({j + 1} of {n}) Writing metadata to {image_path.split("/")[-1]}...')
                        for i, face in enumerate(faces):
                            # use exiftool to save metadata to files
                            with exiftool.ExifToolHelper() as et:
                                tags = et.get_tags(files=image_path,
                                                   tags=["XMP:RegionName", "XMP:RegionType", "XMP:PersonInImage"])[0]
                                # st.write(tags)
                                if "XMP:PersonInImage" not in tags:
                                    # st.write(f'Setting PersonInImage to {person_shown}')
                                    et.execute(f"-XMP:PersonInImage={face.person_shown}", image_path)
                                elif face.person_shown not in tags["XMP:PersonInImage"]:
                                    # st.write(f'Appending {person_shown} to PersonInImage')
                                    et.execute(f"-XMP:PersonInImage+={face.person_shown}", image_path)
                                if "XMP:RegionName" not in tags:
                                    # st.write(f'Setting RegionName to {person_shown}')
                                    execution_string = str("-XMP-mwg-rs:RegionInfo={AppliedToDimensions={"
                                                           f"W={face.img_width}, H={face.img_height}, "
                                                           "Unit=pixel}, RegionList=[{Area={"
                                                           f"W={face.W}, H={face.H}, X={face.X}, Y={face.Y},"
                                                           "Unit=normalized}, "
                                                           f"Name={face.person_shown},"
                                                           "Type=Face}]}")
                                    print(execution_string)
                                    et.execute(execution_string, image_path)
                                elif face.person_shown not in tags["XMP:RegionName"]:
                                    # st.write(f'Appending {person_shown} to RegionName')
                                    execution_string = str("-XMP-mwg-rs:RegionList+=[{Area={"
                                                           f"W={face.W}, H={face.H}, X={face.X}, Y={face.Y},"
                                                           "Unit=normalized}, "
                                                           f"Name={face.person_shown},"
                                                           "Type=Face}]}")
                                    print(execution_string)
                                    et.execute(execution_string, image_path)
                        j += 1
                    status_bar.empty()
                    st.success("Metadata saved to files! Workflow complete!", icon='✅')

                elif export_metadata:
                    # open/create a csv file for writing
                    with open(os.path.join(IMG_DIR, select_folder, 'metadata.csv'), 'w', newline='') as csv_file:
                        # initialize a csv writer object from the csv module
                        csv_writer = csv.writer(csv_file, dialect='excel', delimiter=',')
                        # define column names for the csv file
                        csv_columns = ['filename', 'region_w', 'region_h', 'region_x', 'region_y', 'person_shown']
                        # write column names to the first row of the csv file
                        csv_writer.writerow(csv_columns)
                        # iterate over the labeled data and write a row to the csv file for each face
                        for fp, faces in sess['labeled'].items():
                            filename = fp.split('/')[-1]
                            for face in faces:
                                csv_writer.writerow([filename, face.W, face.H, face.X, face.Y, face.person_shown])

                    st.success("Metadata exported to csv file! Workflow complete!", icon='✅')
