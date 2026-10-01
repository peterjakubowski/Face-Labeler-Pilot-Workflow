# Face Labeler Image Viewer is a Python-based photography workflow tool
# for viewing tagged people shown in images using Exiftool.
#
# Author: Peter Jakubowski
# Date: 5/9/2024
# Description: Streamlit app that opens a selected folder of images
# and displays images with bounding boxes and names from
# embedded metadata extracted using Exiftool.
#

import streamlit as st

from utils.helpers import list_folders_in_watch_folder, run_image_viewer_workflow


def streamlit_viewer_app():

    st.title("Image Viewer")

    st.write("Extracts embedded metadata from images and displays labeled faces surrounded by bounding boxes.")
    # list all the folders inside the watch folder
    # folder_names = [folder for folder in os.listdir(IMG_DIR) if not folder.startswith(".")]
    folder_names = list_folders_in_watch_folder()
    # Display a warning if there are no subfolders in the 'watch_folder'
    if not folder_names:
        st.warning(
            "The watch folder is empty. Add a folder of images to the watch folder to begin."
        )
    # choose a folder with the streamlit select box
    select_folder = st.selectbox(label='Choose a folder of images to view.',
                                 options=folder_names,
                                 accept_new_options=False,
                                 index=None,
                                 placeholder='Choose a folder of images')
    # click the button to display annotated images
    annotate_faces = st.button(label="View Images")

    if select_folder and annotate_faces:

        run_image_viewer_workflow(select_folder)


streamlit_viewer_app()
