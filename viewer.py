import streamlit as st
import exiftool
import cv2
import os
from imutils import paths
from config import IMG_DIR, IMG_PREVIEW_WIDTH


def streamlit_viewer_app():

    # list all the folders inside the watch folder
    folder_names = [folder for folder in os.listdir(IMG_DIR) if not folder.startswith(".")]
    # choose a folder with the streamlit select box
    select_folder = st.selectbox(label='Choose a folder of images to view and label all faces.',
                                 options=folder_names)
    #
    annotate_faces = st.button(label="Annotate Faces")

    if annotate_faces:
        # list file paths for all images in the selected folder
        image_paths = list(paths.list_images(str(IMG_DIR) + "/" + select_folder))
        # read metadata from all images using exiftool
        with exiftool.ExifToolHelper() as et:

            metadata = et.get_metadata(image_paths)

        for m in metadata:
            if "SourceFile" in m:
                img = cv2.imread(m["SourceFile"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_height, img_width = img.shape[:2]
                height = int((img_height / img_width) * IMG_PREVIEW_WIDTH)
                # st.write(img_width, img_height)
                # st.write(WIDTH, height)
                img = cv2.resize(img, (IMG_PREVIEW_WIDTH, height), cv2.INTER_AREA)

                if "XMP:RegionType" in m:
                    if type(m["XMP:RegionType"]) == str:
                        st.write('string')
                        w = int(m["XMP:RegionAreaW"] * height)
                        h = int(m["XMP:RegionAreaH"] * IMG_PREVIEW_WIDTH)
                        x = int(m["XMP:RegionAreaX"] * height)
                        y = int(m["XMP:RegionAreaY"] * IMG_PREVIEW_WIDTH)
                        person_shown = m["XMP:RegionName"]
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        # get text size
                        text_size = cv2.getTextSize(person_shown, cv2.FONT_HERSHEY_PLAIN, 1.3, 2)
                        dim = text_size[0]
                        baseline = text_size[1]
                        # Use text size to create a black rectangle
                        cv2.rectangle(img, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0),
                                      cv2.FILLED)
                        # put text labels on the image
                        cv2.putText(img, person_shown, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)

                    elif type(m["XMP:RegionType"]) == list:
                        # st.write('list')
                        for i in range(len(m["XMP:RegionType"])):
                            if m["XMP:RegionType"][i] == 'Face':
                                w = int(m["XMP:RegionAreaW"][i] * height)
                                h = int(m["XMP:RegionAreaH"][i] * IMG_PREVIEW_WIDTH)
                                x = int(m["XMP:RegionAreaX"][i] * height)
                                y = int(m["XMP:RegionAreaY"][i] * IMG_PREVIEW_WIDTH)
                                person_shown = m["XMP:RegionName"][i]
                                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
                                # get text size
                                text_size = cv2.getTextSize(person_shown, cv2.FONT_HERSHEY_PLAIN, 1.3, 2)
                                dim = text_size[0]
                                baseline = text_size[1]
                                # Use text size to create a black rectangle
                                cv2.rectangle(img, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0),
                                             cv2.FILLED)
                                # put text labels on the image
                                cv2.putText(img, person_shown, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)

                st.image(img, width=800)


streamlit_viewer_app()
