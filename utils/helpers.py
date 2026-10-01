from collections import defaultdict
from pathlib import Path

import streamlit as st
from image_utils import list_image_paths

from config import IGNORE_FACE_TEXT, IMG_DIR, IMG_PREVIEW_WIDTH
from utils.exiftool import extract_metadata_from_files_with_exiftool
from utils.face_classifier import face_conn
from utils.image_processing import (
    annotate_image_with_face_region_using_opencv,
    detect_faces,
    prepare_image_for_inference,
)


def list_folders_in_watch_folder() -> list[str]:
    # Make the 'watch_folder' directory if it does not exist
    Path.mkdir(IMG_DIR, exist_ok=True)
    # List the subfolders of the 'watch_folder'
    # folder_names = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    folder_names = [d.name for d in IMG_DIR.iterdir() if d.is_dir()]

    return folder_names


def run_face_detection_workflow(select_folder: str):
    # list all the images (paths) in the selected folder
    st.session_state["image_paths"] = sorted(
        list_image_paths(IMG_DIR / select_folder), key=lambda x: x.name
    )
    # detect faces in all the images, get a list/queue of faces (instances of Face class)
    st.session_state["faces_detected"] = detect_faces(
        img_paths=st.session_state.image_paths
    )
    # count how many faces were detected
    st.session_state["faces_count"] = len(st.session_state.faces_detected)
    # if we didn't detect any faces, delete the queue from the session state and display a message
    if st.session_state.faces_count < 1:
        del st.session_state["faces_detected"]
        st.warning(
            f"Workflow complete! "
            f"Looked for faces in {len(st.session_state['image_paths'])} images "
            f"and {st.session_state['faces_count']} faces were found."
        )
    # count of faces labeled
    st.session_state["face_i"] = 1
    # dictionary of labeled faces
    st.session_state["labeled"] = defaultdict(list)
    # dictionary of names/identities and counts
    st.session_state["name_options"] = defaultdict(int)


def record_name(selected_name: str) -> None:
    """
    Function for recording a name for a labeled person.
    Updates the current face class with modifications.
    If the current face is labeled, then it is added to
    the dictionary of labeled faces and removed from
    the queue of faces to label.
    :return: None
    """

    if selected_name:
        # peek at the first face in the queue of detected faces
        current_face = st.session_state["faces_detected"][0]

        # ignore this face if the selected name is set to ignore
        if selected_name == IGNORE_FACE_TEXT:
            # decrement the count of detected faces
            st.session_state.faces_count -= 1
        else:
            # update the current face's person shown attribute with the selected name
            current_face.person_shown = selected_name
            # increment the count for the number of times faces have been labeled with this name
            st.session_state.name_options[current_face.person_shown] += 1
            st.session_state.face_i += 1
            # add the current face to the dictionary of labeled faces
            st.session_state.labeled[str(current_face.img_path)].append(current_face)
            # if the current face has an encoding, append it along with the name
            # to the list of encodings and names for future face recognitions
            if len(current_face.encoding) > 0 and not face_conn.is_in(
                current_face.encoding[0]
            ):
                # add new face to the face classifier if we don't already have a similar record
                face_conn.add_new_face(
                    current_face.person_shown, current_face.encoding[0]
                )

        # pop the current face from the queue
        st.session_state["faces_detected"].popleft()


def run_image_viewer_workflow(select_folder: str):

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
            _, img = prepare_image_for_inference(
                image_path=Path(source_file), img_size=IMG_PREVIEW_WIDTH
            )
            # get the open image's width and height
            width = img.shape[1]
            height = img.shape[0]
            # iterate over each region and annotate the image if region is 'Face'
            for i in range(len(region_type)):
                if region_type[i] == "Face":
                    img = annotate_image_with_face_region_using_opencv(
                        img=img,
                        person_shown=region_name[i],
                        w=int(region_area_w[i] * height),
                        h=int(region_area_h[i] * width),
                        x=int(region_area_x[i] * height),
                        y=int(region_area_y[i] * width),
                    )

            # display the annotated image
            st.image(img)
