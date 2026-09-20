from pathlib import Path

import face_recognition
import numpy as np
from image_utils import list_image_paths

from config import LIB_DIR
from models.face import Face
from utils.exiftool import extract_metadata_from_files_with_exiftool
from utils.image_processing import prepare_image_for_inference


def prepare_reference_data() -> list[dict]:
    """
    Prepares the reference data for loading into the face classifier.
    Looks in the `library` folder for paths to images and extracts their metadata
    using exiftool. Reads the metadata and checks for XMP:Regions of type 'Face'.
    Loads each image and feeds every face to the face recognition where a face
    encoding/embedding is generated. Returns a dictionary with the name or
    the person shown, taken from XMP:RegionName, paired with the embedding.

    example return dict:
    {"name": "Person shown name", "embedding": np.array(128)}
    :return: List of dictionaries with names of persons shown and their face embedding.
    """

    # make the library directory if it doesn't exist
    Path.mkdir(LIB_DIR, exist_ok=True)
    # list all the image paths from the library directory
    library_image_paths = list_image_paths(LIB_DIR)
    # extract all the metadata from image files
    library_image_metadata = extract_metadata_from_files_with_exiftool(library_image_paths)
    # keeps a list of faces that we find in the xmp/mwg regions
    library_faces: list[Face] = []
    # iterate over each image file's metadata
    for m in library_image_metadata:
        # check if there's a region in the xmp
        if 'XMP:RegionType' in m:
            # extract the file's path from the metadata
            source_file = Path(m.get('SourceFile'))
            # open the image
            shape, img = prepare_image_for_inference(Path(source_file))
            # get lists of region type, names, and bounding box coordinates
            region_type = list(region_type if isinstance(region_type := m.get('XMP:RegionType'), list) else [region_type])
            persons_shown = list(persons_shown if isinstance(persons_shown := m.get('XMP:RegionName'), list) else [persons_shown])
            region_area_w = list(region_area_w if isinstance(region_area_w := m.get('XMP:RegionAreaW'), list) else [region_area_w])
            region_area_h = list(region_area_h if isinstance(region_area_h := m.get('XMP:RegionAreaH'), list) else [region_area_h])
            region_area_x = list(region_area_x if isinstance(region_area_x := m.get('XMP:RegionAreaX'), list) else [region_area_x])
            region_area_y = list(region_area_y if isinstance(region_area_y := m.get('XMP:RegionAreaY'), list) else [region_area_y])
            # iterate over the regions and check if it's a face
            for i, t in enumerate(region_type):
                if t == 'Face':
                    # scale the bounding box coordinates to the open image's size
                    w = int(np.round(region_area_w[i] * img.shape[0], 0))
                    h = int(np.round(region_area_h[i] * img.shape[1], 0))
                    x = int(np.round(region_area_x[i] * img.shape[0], 0))
                    y = int(np.round(region_area_y[i] * img.shape[1], 0))
                    # convert the bounding box to top, right, bottom, left
                    top = y
                    right = x + w
                    bottom = y + h
                    left = x
                    # keep our face location as a tuple
                    face_location = (top, right, bottom, left)
                    # feed the open image along with the face location to face recognition to generate a face encoding
                    encodings = face_recognition.face_encodings(
                        face_image=img,
                        known_face_locations=[face_location],
                        num_jitters=1,
                        model="large"
                    )
                    # construct our face object with image path, size, face location and encoding
                    new_face = Face(
                        img_path=source_file,
                        img_width=shape[1],
                        img_height=shape[0],
                        img_resized_width=img.shape[1],
                        img_resized_height=img.shape[0],
                        face_location=face_location,
                        encoding=encodings
                    )
                    # name the face with the persons shown metadata
                    new_face.person_shown = persons_shown[i]
                    # add the face to the list of faces in the library
                    library_faces.append(new_face)
    # process the library faces for reference data
    reference_data: list[dict] = []
    # iterate over each face and keep the name and embedding in a dict
    for face in library_faces:
        reference_data.append(
            {"name": face.person_shown,
             "embedding": face.encoding[0]}
        )
    # return the processed reference data
    return reference_data
