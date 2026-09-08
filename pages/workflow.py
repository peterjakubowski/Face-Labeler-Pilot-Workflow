# Face Labeler Pilot is a Python-based photography workflow tool
# for tagging people shown in images using face recognition.
#
# Author: Peter Jakubowski
# Date: 5/9/2024
# Description: Streamlit app that opens a selected folder of images
# and detects faces for labeling.
#
#

import time

import face_recognition
import pandas as pd
import streamlit as st

from config import AUTO_CONFIRM_MATCHES_TIME, COMPARE_FACES_TOLERANCE
from utils.csv import export_metadata_to_csv
from utils.exiftool import write_metadata_with_exiftool
from utils.helpers import (
    list_folders_in_watch_folder,
    record_name,
    run_face_detection_workflow,
)


def streamlit_workflow_app():

    #       ==========================================
    # INFO: ===== Face Labeler Pilot Introduction ====
    #       ==========================================

    st.title("Face Labeler Pilot")
    intro_text = ("Face Labeler Pilot is a 3-step post-production workflow tool "
                  "that uses face recognition to tag people shown in photographs.")
    st.markdown(intro_text)

    #       =====================================
    # INFO: ===== Begin Step 1: Detect Faces ====
    #       =====================================

    st.subheader("Step 1: Detect Faces", divider="gray")

    # List the subfolders of the 'watch_folder'
    folder_names = list_folders_in_watch_folder()
    # Display a warning if there are no subfolders in the 'watch_folder'
    if not folder_names:
        st.warning("The watch folder is empty. Add a folder of images to the watch folder to begin.")
    # Streamlit select widget, gives the user a way to select a folder of images
    select_folder = st.selectbox(label='Choose a folder of images to scan for faces',
                                 index=None,
                                 options=folder_names,
                                 placeholder='Choose a folder of images',
                                 label_visibility='collapsed',
                                 accept_new_options=False
                                 )

    if select_folder:
        # Streamlit button widget, kicks off the face detection workflow when pressed
        start_face_detection = st.button(label="Detect Faces")
        if start_face_detection:

            #       ==========================================
            # INFO: ===== Run face detection workflow:    ====
            #       ===== Scan all images in the selected ====
            #       ===== folder and detect all faces.    ====
            #       ==========================================

            run_face_detection_workflow(select_folder)

    if 'faces_detected' in st.session_state:
        success_text = (f"Face detection is complete! "
                        f"Found {st.session_state.faces_count} faces "
                        f"in {len(st.session_state.image_paths)} images."
                        )
        st.success(success_text, icon='✅')

        #       ====================================
        # INFO: ===== Begin Step 2: Label Faces ====
        #       ====================================

        st.subheader("Step 2: Label Faces", divider="gray")

        # check if there are faces in our queue
        if st.session_state['faces_detected']:
            # ask the user if the workflow should automatically confirm/accept matches
            auto_confirm_matches = st.checkbox(label="Auto confirm matches?",
                                               value=False,
                                               key='auto_confirm_matches')
            status_bar = st.progress(st.session_state.face_i / st.session_state.faces_count,
                                     text=f'Labeling face {st.session_state.face_i} of {st.session_state.faces_count}')
            # pop the next face from the queue
            current_face = st.session_state['faces_detected'][0]
            # open cropped image of current face
            current_face_img = current_face.open_face_image()
            # check if the current face has an encoding
            if len(current_face.encoding) > 0:

                #       ==============================================
                # INFO: ===== Begin face recognition:             ====
                #       ===== Compare current face to known faces ====
                #       ==============================================

                # compare the face encoding to existing encodings to see if we can find a match
                # note: the lower the tolerance, the more sensitive the algorithm is at matching faces
                matches = face_recognition.compare_faces(st.session_state.data['encodings'],
                                                         current_face.encoding[0],
                                                         tolerance=COMPARE_FACES_TOLERANCE)

                if True in matches:
                    # count matches and find the name with the most matches
                    matched_indices = [i for (i, b) in enumerate(matches) if b]
                    count = {}
                    for i in matched_indices:
                        name = st.session_state.data['names'][i]
                        count[name] = count.get(name, 0) + 1
                    predicted_name = max(count, key=count.get)

                    # if auto confirm matches is not checked, then provide a form to label the current face
                    if not auto_confirm_matches:
                        with st.form(key="predicted_name_form", clear_on_submit=True):
                            # display a thumbnail of the current face
                            st.image(current_face_img, width=100)

                            st.write(f'I think this face belongs to **{predicted_name}**, can you confirm?')
                            selected_name = st.selectbox(label=('The predicted name has been pre-selected, '
                                                                'click the submit button to confirm.\n\n'
                                                                'Select "Not a face" to skip this face.\n\n'
                                                                'Or, add or select someone else.\n'),
                                                         options=['Not a face'] + sorted(
                                                             st.session_state.name_options.keys()),
                                                         index=sorted(st.session_state.name_options.keys()).index(
                                                             predicted_name) + 1,
                                                         accept_new_options=True,
                                                         placeholder=None)

                            submitted = st.form_submit_button(label='Submit')
                            if submitted:
                                record_name(selected_name=selected_name)
                                st.rerun()

                    # if auto confirm matches is checked, then label the current face with the predicted name
                    elif auto_confirm_matches:
                        # display a thumbnail of the current face
                        st.image(current_face_img, width=100)
                        st.write(f"This face belongs to **{predicted_name}**")
                        st.selectbox(label="Predicted name",
                                     options=sorted(st.session_state.name_options.keys()),
                                     index=sorted(st.session_state.name_options.keys()).index(predicted_name),
                                     disabled=True
                                     )
                        # wait for a moment, user can still interrupt by unchecking auto confirm matches
                        time.sleep(AUTO_CONFIRM_MATCHES_TIME)
                        record_name(selected_name=predicted_name)
                        st.rerun()

                else:
                    with st.form(key="new_face_form", clear_on_submit=True):
                        # display a thumbnail of the current face
                        st.image(current_face_img, width=100)
                        st.write("I don't recognize this face, who is this?")
                        selected_name = st.selectbox(label=('Type in a new name or select one from the list. '
                                                            'Select "Not a face" to skip this face.'),
                                                     options=['Not a face'] + sorted(
                                                         st.session_state.name_options.keys()),
                                                     accept_new_options=True,
                                                     placeholder=None,
                                                     index=None)

                        submitted = st.form_submit_button(label='Submit')
                        if submitted:
                            record_name(selected_name=selected_name)
                            st.rerun()

            elif not current_face.encoding:
                with st.form(key="no_face_encoding_form", clear_on_submit=True):
                    # display a thumbnail of the current face
                    st.image(current_face_img, width=100)
                    st.write('This face has no encoding. Is this a face?')
                    selected_name = st.selectbox(label=('Type in a new name or select one from the list. '
                                                        'Select "Not a face" to skip this face.'),
                                                 options=['Not a face'] + sorted(st.session_state.name_options.keys()),
                                                 accept_new_options=True,
                                                 placeholder=None,
                                                 index=0
                                                 )

                    submitted = st.form_submit_button(label='Continue')
                    if submitted:
                        record_name(selected_name=selected_name)
                        st.rerun()

        # if our queue of faces is empty, check if we have labeled any images
        if not st.session_state['faces_detected']:
            # if we have labeled data, let's embed the face locations and names in the image metadata
            if 'labeled' in st.session_state:
                if not st.session_state['labeled']:
                    st.success(f'{len(st.session_state.labeled)} faces were labeled. Workflow complete!',
                               icon='✅')
                elif st.session_state['labeled']:
                    st.success('All faces have been labeled!', icon='✅')

                    # display a dataframe with counts of unique names/labels
                    df = pd.DataFrame(data=st.session_state.name_options.items(),
                                      columns=['names', 'counts'])
                    df.set_index('names', inplace=True)
                    st.dataframe(df.sort_index())

                    #       ==================================================
                    # INFO: ===== Begin Step 3: Write/Save/Embed Metadata ====
                    #       ==================================================

                    st.subheader("Step 3: Save Metadata", divider="gray")

                    # created columns for buttons to display side-by-side
                    col1, col2, _, _ = st.columns(4)
                    with col1:
                        write_metadata_button = st.button(label="Write Metadata")
                    with col2:
                        export_metadata_button = st.button(label="Export Metadata")

                    if write_metadata_button:
                        # write/embed metadata to original files using exiftool
                        write_metadata_with_exiftool()

                        st.success("Metadata saved to files! Workflow complete!", icon='✅')

                    elif export_metadata_button:
                        # export metadata to a csv file next to original files
                        export_metadata_to_csv(select_folder)

                        st.success("Metadata exported to csv file! Workflow complete!", icon='✅')


streamlit_workflow_app()  # Run the Streamlit app
