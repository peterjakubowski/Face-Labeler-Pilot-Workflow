# Face Labeler Pilot is a Streamlit and Python-based photography workflow tool
# for tagging and viewing people shown in images.
#
# Author: Peter Jakubowski
# Date: 5/9/2024
# Description: Streamlit app that facilitates tagging persons shown in images.
#
#

import streamlit as st

workflow = st.Page(page="pages/workflow.py", title="Workflow")

viewer = st.Page(page="pages/viewer.py", title="Viewer")

settings = st.Page(page="pages/settings.py", title="Settings")

pages = st.navigation([workflow, viewer, settings])

pages.run()
