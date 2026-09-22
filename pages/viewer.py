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

    st.write("Extracts embedded metadata from images and displays labeled faces surrounded by bounding boxes.")
    # list all the folders inside the watch folder
    # folder_names = [folder for folder in os.listdir(IMG_DIR) if not folder.startswith(".")]
    folder_names = list_folders_in_watch_folder()
    # choose a folder with the streamlit select box
    select_folder = st.selectbox(label='Choose a folder of images to view.',
                                 options=folder_names,
                                 accept_new_options=False,
                                 index=None,
                                 placeholder=None)
    # click the button to display annotated images
    annotate_faces = st.button(label="View Images")

    if select_folder and annotate_faces:
        # list file paths for all images in the selected folder limit to 50
        image_paths = list(list_image_paths(IMG_DIR / select_folder))[:50]
        # read metadata from all images using exiftool
        metadata = extract_metadata_from_files_with_exiftool(image_paths)
        # iterate through each image metadata
        for m in metadata:
            # check if our metadata has a path to the source file
            source_file: str | None = m.get("SourceFile", None)
            # retrieve region metadata: type, name, area(w, h, x, y)
            region_type: list[str] = (
                region_type
                if isinstance(region_type := m.get("XMP:RegionType", []), list)
                else [region_type]
            )
            region_name: list[str] = (
                region_name
                if isinstance(region_name := m.get("XMP:RegionName", []), list)
                else [region_name]
            )
            region_area_w: list[float] = (
                region_area_w
                if isinstance(region_area_w := m.get("XMP:RegionAreaW", []), list)
                else [region_area_w]
            )
            region_area_h: list[float] = (
                region_area_h
                if isinstance(region_area_h := m.get("XMP:RegionAreaH", []), list)
                else [region_area_h]
            )
            region_area_x: list[float] = (
                region_area_x
                if isinstance(region_area_x := m.get("XMP:RegionAreaX", []), list)
                else [region_area_x]
            )
            region_area_y: list[float] = (
                r_area_y
                if isinstance(r_area_y := m.get("XMP:RegionAreaY", []), list)
                else [r_area_y]
            )

            if source_file is not None:
                # open an image to annotate
                _, img = prepare_image_for_inference(image_path=Path(source_file), img_size=IMG_PREVIEW_WIDTH)
                # get the open image's width and height
                width = img.shape[1]
                height = img.shape[0]
                # iterate over each region and annotate the image if region is 'Face'
                for i in range(len(region_type)):
                    if region_type[i] == 'Face':
                        img = annotate_image_with_face_region_using_opencv(
                            img=img,
                            person_shown=region_name[i],
                            w=int(region_area_w[i] * height),
                            h=int(region_area_h[i] * width),
                            x=int(region_area_x[i] * height),
                            y=int(region_area_y[i] * width)
                        )

                # display the annotated image
                st.image(img)


streamlit_viewer_app()
