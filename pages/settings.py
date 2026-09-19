import streamlit as st

from config import COMPARE_FACES_TOLERANCE, TOP_K
from utils.face_classifier import face_conn
from utils.library import prepare_reference_data

st.title("Face recognition settings")

st.write(st.session_state)

top_k_input = st.number_input(
    label="Top K",
    min_value=1,
    max_value=10,
    step=1,
    value=st.session_state.get('top_k', TOP_K),
    help=(
        "Controls how many close matches the AI reviews before applying a name tag. "
        "Lower values (1–3) work best if you have very few reference photos of a person. "
        "Higher values require a larger consensus among your existing collection, which "
        "improves tagging accuracy but requires you to have already tagged multiple photos of that person."
    )
)

threshold_input = st.number_input(
    label="Threshold",
    min_value=0.10,
    max_value=1.0,
    step=0.05,
    value=st.session_state.get('threshold', COMPARE_FACES_TOLERANCE),
    help=(
        "Controls how aggressively the AI applies automatic tags.\n\n"
        "• Lower (0.3 – 0.45): High accuracy. "
        "Prevents wrong tags, but forces you to manually tag photos the AI wasn't 100% sure about.\n\n"
        "• Higher (0.65 – 0.8): Auto-tag expansion. "
        "The AI will aggressively guess and tag more faces, but you may occasionally have to fix a mistagged photo."
    )
)

if top_k_input != st.session_state.get('top_k', TOP_K) or threshold_input != st.session_state.get('threshold', COMPARE_FACES_TOLERANCE):
    save_settings_button = st.button("Save settings")

    if save_settings_button:
        st.session_state['top_k'] = int(top_k_input)
        st.session_state['threshold'] = float(threshold_input)
        st.rerun()

if not face_conn.is_initialized():
    initialize_button = st.button("Initialize Face Classifier", help="Load labeled faces from image library folder.")
    if initialize_button:
        with st.spinner(text="Initializing face classifier", show_time=True):
            reference_data = prepare_reference_data()
            face_conn.load_reference_data(reference_data)
            st.rerun()

else:
    st.write(face_conn.info())

    reset_button = st.button("Reset Face Classifier")
    if reset_button:
        face_conn.reset()
        if 'top_k' in st.session_state:
            del st.session_state['top_k']
        if 'threshold' in st.session_state:
            del st.session_state['threshold']
        st.rerun()
