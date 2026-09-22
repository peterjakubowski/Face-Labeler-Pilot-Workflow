# Face Recognition Settings
#
# Author: Peter Jakubowski
# Date: 9/19/2026
# Description: Streamlit app page that controls how
# the app recognizes faces.
#
#

import streamlit as st

from config import COMPARE_FACES_TOLERANCE, TOP_K
from utils.face_classifier import face_conn
from utils.library import prepare_reference_data

st.title("Face Recognition Settings")

st.write("Control how the system recognizes faces.")

top_k_input = st.number_input(
    label="Top K",
    min_value=1,
    max_value=10,
    step=1,
    value=st.session_state.get('top_k', TOP_K),
    help="Controls how many close matches the system reviews before predicting a name."
)

threshold_input = st.number_input(
    label="Threshold",
    min_value=0.10,
    max_value=1.0,
    step=0.05,
    value=st.session_state.get('threshold', COMPARE_FACES_TOLERANCE),
    help="Controls how aggressively the system applies automatic tags."
)

st.info(face_conn.info())

if top_k_input != st.session_state.get('top_k', TOP_K) or threshold_input != st.session_state.get('threshold', COMPARE_FACES_TOLERANCE):
    save_settings_button = st.button("Save settings", type="secondary")

    if save_settings_button:
        st.session_state['top_k'] = int(top_k_input)
        st.session_state['threshold'] = float(threshold_input)
        st.rerun()

if not face_conn.is_initialized():
    initialize_button = st.button(
        "Load Faces Library",
        type="primary",
        help="Scans your library folder to extract existing face region metadata and names from labeled images."
    )
    if initialize_button:
        with st.spinner(text="Initializing face classifier", show_time=True):
            reference_data = prepare_reference_data()
            face_conn.load_reference_data(reference_data)
            st.rerun()

else:
    with st.popover(
        label="Reset Face Classifier",
        help=(
            "Clears all loaded name metadata and face embedding from the application's memory, "
            "resetting the classifier to an uninitialized state."
        )
    ):
        st.write("Are you sure you want to reset the face classifier?\n\n"
                 "This will clear all loaded name metadata and face embeddings.")
        reset_button = st.button(
            "Confirm Reset",
            type="secondary",

        )
        if reset_button:
            face_conn.reset()
            if 'top_k' in st.session_state:
                del st.session_state['top_k']
            if 'threshold' in st.session_state:
                del st.session_state['threshold']
            st.rerun()
