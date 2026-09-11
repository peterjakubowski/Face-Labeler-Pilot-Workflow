import time
from pathlib import Path

import exiftool
import streamlit as st

from models.face import Face


def extract_metadata_from_files_with_exiftool(image_paths: list[Path]) -> list[dict]:
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(image_paths)

    return metadata


def get_tags_with_exiftool(image_path: Path) -> dict:
    with exiftool.ExifToolHelper() as et:
        tags = et.get_tags(
            files=image_path,
            tags=["XMP:RegionName", "XMP:RegionType", "XMP:PersonInImage"]
        )
    return tags[0]


def add_person_shown_with_exiftool(image_path: Path, face: Face):
    with exiftool.ExifToolHelper() as et:
        et.execute(f"-XMP:PersonInImage={face.person_shown}", image_path)


def append_person_shown_with_exiftool(image_path: Path, face: Face):
    with exiftool.ExifToolHelper() as et:
        et.execute(f"-XMP:PersonInImage+={face.person_shown}", image_path)


def add_region_with_exiftool(image_path: Path, face: Face):
    execution_string = str("-XMP-mwg-rs:RegionInfo={AppliedToDimensions={"
                           f"W={face.img_width}, H={face.img_height}, "
                           "Unit=pixel}, RegionList=[{Area={"
                           f"W={face.W}, H={face.H}, X={face.X}, Y={face.Y},"
                           "Unit=normalized}, "
                           f"Name={face.person_shown},"
                           "Type=Face}]}")
    with exiftool.ExifToolHelper() as et:
        et.execute(execution_string, image_path)


def append_region_with_exiftool(image_path: Path, face: Face):
    execution_string = str("-XMP-mwg-rs:RegionList+=[{Area={"
                           f"W={face.W}, H={face.H}, X={face.X}, Y={face.Y},"
                           "Unit=normalized}, "
                           f"Name={face.person_shown},"
                           "Type=Face}]}")
    with exiftool.ExifToolHelper() as et:
        et.execute(execution_string, image_path)


def write_metadata_with_exiftool():
    status_bar = st.progress(0, 'Begin writing metadata to files!')
    time.sleep(1)
    n = len(st.session_state.labeled)
    for j, (image_path, faces) in enumerate(st.session_state.labeled.items()):
        # update our progress
        status_bar.progress(
            (j + 1) / n,
            text=f'({j + 1} of {n}) Writing metadata to {image_path.name}'
        )

        for face in faces:
            # use exiftool to save metadata to files
            tags = get_tags_with_exiftool(image_path)
            if "XMP:PersonInImage" not in tags:
                add_person_shown_with_exiftool(image_path, face)
            elif face.person_shown not in set(tags["XMP:PersonInImage"]):
                append_person_shown_with_exiftool(image_path, face)
            if "XMP:RegionName" not in tags:
                add_region_with_exiftool(image_path, face)
            elif face.person_shown not in set(tags["XMP:RegionName"]):
                append_region_with_exiftool(image_path, face)

    status_bar.empty()
