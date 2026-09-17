import time

import streamlit as st

from config import COMPARE_FACES_TOLERANCE, TOP_K, reference_data
from utils.face_classifier import face_conn

st.header("Face classifier settings")

st.write(st.session_state)

top_k_input = st.number_input(
    label="Top K",
    min_value=1,
    max_value=10,
    step=1,
    value=st.session_state.get('top_k', TOP_K),
    help="Top k nearest neighbors"
)

threshold_input = st.number_input(
    label="Threshold",
    min_value=0.10,
    max_value=1.0,
    step=0.05,
    value=st.session_state.get('threshold', COMPARE_FACES_TOLERANCE)
)

if top_k_input != st.session_state.get('top_k') or threshold_input != st.session_state.get('threshold'):
    save_settings_button = st.button("Save settings")

    if save_settings_button:
        st.session_state['top_k'] = int(top_k_input)
        st.session_state['threshold'] = float(threshold_input)
        st.rerun()

if not face_conn.is_initialized():
    initialize_button = st.button("Initialize")
    if initialize_button:
        with st.spinner("Initializing face classifier"):
            time.sleep(1)
            face_conn.load_reference_data(reference_data)
            st.rerun()

else:
    select_options = [reference_data[i].get('name') for i in range(5)]

    select_name = st.selectbox(label="select_box", options=select_options)

    select_name_index = int(select_name.split(" ")[-1]) - 1

    test_embedding = reference_data[select_name_index].get("embedding") * 0.95

    prediction, confidence = face_conn.predict(
        test_embedding,
        k=st.session_state.get('top_k', TOP_K),
        threshold=st.session_state.get('threshold', COMPARE_FACES_TOLERANCE))

    st.write(prediction)

    st.write(confidence)

    reset_button = st.button("Reset")
    if reset_button:
        face_conn.reset()
        del st.session_state['top_k']
        del st.session_state['threshold']
        st.rerun()
