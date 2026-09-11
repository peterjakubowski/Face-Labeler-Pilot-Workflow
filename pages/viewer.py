# Face Labeler Image Viewer is a Python-based photography workflow tool
# for viewing tagged people shown in images using Exiftool.
#
# Author: Peter Jakubowski
# Date: 5/9/2024
# Description: Streamlit app that opens a selected folder of images
# and displays images with bounding boxes and names from
# embedded metadata extracted using Exiftool.
#
#

from pathlib import Path

import streamlit as st
from image_utils import list_image_paths

from config import IMG_DIR, IMG_PREVIEW_WIDTH
from utils.exiftool import extract_metadata_from_files_with_exiftool
from utils.helpers import list_folders_in_watch_folder
from utils.image_processing import (
    annotate_image_with_face_region_using_opencv,
    prepare_image_for_inference,
)


def streamlit_viewer_app():

    st.title("Image Viewer")

    # list all the folders inside the watch folder
    # folder_names = [folder for folder in os.listdir(IMG_DIR) if not folder.startswith(".")]
    folder_names = list_folders_in_watch_folder()
    # choose a folder with the streamlit select box
    select_folder = st.selectbox(label='Choose a folder of images to view.',
                                 options=folder_names,
                                 accept_new_options=False,
                                 index=None,
                                 placeholder="Choose a folder of images")
    # click the button to display annotated images
    annotate_faces = st.button(label="View Annotated Images")

    if annotate_faces:
        # list file paths for all images in the selected folder limit to 50
        image_paths = list(list_image_paths(str(IMG_DIR) + "/" + select_folder))[:50]
        # read metadata from all images using exiftool
        metadata = extract_metadata_from_files_with_exiftool(image_paths)
        # iterate through each image metadata
        for m in metadata:
            # check if our metadata has a path to the source file
            if "SourceFile" in m:
                # open an image to annotate
                _, img = prepare_image_for_inference(image_path=Path(m["SourceFile"]), img_size=IMG_PREVIEW_WIDTH)
                width = img.shape[1]
                height = img.shape[0]
                # check if there is a region to annotate
                if "XMP:RegionType" in m:
                    # if our region type is a str, there is one region
                    if isinstance(m["XMP:RegionType"], str) and m["XMP:RegionType"] == 'Face':
                        img = annotate_image_with_face_region_using_opencv(
                            img=img,
                            person_shown=m["XMP:RegionName"],
                            w=int(m["XMP:RegionAreaW"] * height),
                            h=int(m["XMP:RegionAreaH"] * width),
                            x=int(m["XMP:RegionAreaX"] * height),
                            y=int(m["XMP:RegionAreaY"] * width)
                        )
                    # if our region is a list, there are multiple regions
                    elif isinstance(m["XMP:RegionType"], list):
                        # iterate over all regions
                        for i in range(len(m["XMP:RegionType"])):
                            if m["XMP:RegionType"][i] == 'Face':
                                img = annotate_image_with_face_region_using_opencv(
                                    img=img,
                                    person_shown=m["XMP:RegionName"][i],
                                    w=int(m["XMP:RegionAreaW"][i] * height),
                                    h=int(m["XMP:RegionAreaH"][i] * width),
                                    x=int(m["XMP:RegionAreaX"][i] * height),
                                    y=int(m["XMP:RegionAreaY"][i] * width)
                                )
                st.image(img)


streamlit_viewer_app()
