import os
from collections import defaultdict
from pathlib import Path

import streamlit as st
from image_utils import list_image_paths

from config import IMG_DIR, IMG_SIZE
from utils.image_processing import detect_faces


def list_folders_in_watch_folder() -> list[str]:
    # Make the 'watch_folder' directory if it does not exist
    Path.mkdir(IMG_DIR, exist_ok=True)
    # List the subfolders of the 'watch_folder'
    folder_names = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]

    return folder_names


def run_face_detection_workflow(select_folder: str):
    # list all the images (paths) in the selected folder
    st.session_state['image_paths'] = sorted(list_image_paths(os.path.join(IMG_DIR, select_folder)),
                                             key=lambda x: str(x).split('/')[-1])
    # detect faces in all the images, get a list/queue of faces (instances of Face class)
    st.session_state['faces_detected'] = detect_faces(img_paths=st.session_state.image_paths, img_size=IMG_SIZE)
    # count how many faces were detected
    st.session_state['faces_count'] = len(st.session_state.faces_detected)
    # if we didn't detect any faces, delete the queue from the session state and display a message
    if st.session_state.faces_count < 1:
        del st.session_state['faces_detected']
        st.warning(f"Workflow complete! "
                   f"Looked for faces in {len(st.session_state['image_paths'])} images "
                   f"and {st.session_state['faces_count']} faces were found.")
    # count of faces labeled
    st.session_state['face_i'] = 1
    # dictionary of labeled faces
    st.session_state['labeled'] = defaultdict(list)
    # dictionary of face encodings and names
    st.session_state['data'] = {'encodings': [], 'names': []}
    # dictionary of names/identities and counts
    st.session_state['name_options'] = defaultdict(int)


def record_name(selected_name: str) -> None:
    """
    Function for recording a name for a labeled person.
    Updates the current face class with modifications.
    If the current face is labeled, then it is added to
    the dictionary of labeled faces and removed from
    the queue of faces to label.
    :return: None
    """

    if selected_name:
        # peek at the first face in the queue of detected faces
        current_face = st.session_state['faces_detected'][0]

        if selected_name == 'Not a face':
            # decrement the count of detected faces
            st.session_state.faces_count -= 1
        else:
            # update the current face's person shown attribute with the selected name
            current_face.person_shown = selected_name
            # increment the count for the number of times faces have been labeled with this name
            st.session_state.name_options[current_face.person_shown] += 1
            st.session_state.face_i += 1
            # add the current face to the dictionary of labeled faces
            st.session_state.labeled[current_face.img_path].append(current_face)
            # if the current face has an encoding, append it along with the name
            # to the list of encodings and names for future face recognitions
            if len(current_face.encoding) > 0:
                st.session_state.data['encodings'].append(current_face.encoding[0])
                st.session_state.data['names'].append(current_face.person_shown)
        # pop the current face from the queue
        st.session_state['faces_detected'].popleft()
