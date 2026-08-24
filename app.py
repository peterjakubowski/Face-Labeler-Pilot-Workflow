# Face Labeler Labeler Pilot is a Streamlit and Python-based photography workflow tool
# for tagging and viewing people shown in images.
#
# Author: Peter Jakubowski
# Date: 5/9/2024
# Description: Streamlit app that facilitates tagging persons shown in images.
#
#

import streamlit as st

workflow = st.Page(page="pages/workflow.py", title="Face Labeler Workflow")

viewer = st.Page(page="pages/viewer.py", title="Image Viewer")

pages = st.navigation([workflow, viewer])

pages.run()
