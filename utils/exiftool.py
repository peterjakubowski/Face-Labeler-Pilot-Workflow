import time

import exiftool
import streamlit as st


def write_metadata_with_exiftool():
    status_bar = st.progress(0, 'Begin writing metadata to files!')
    time.sleep(1)
    n = len(st.session_state.labeled)
    for j, (image_path, faces) in enumerate(st.session_state.labeled.items()):
        status_bar.progress(
            (j + 1) / n,
            text=f'({j + 1} of {n}) Writing metadata to {image_path.split("/")[-1]}...'
        )
        for i, face in enumerate(faces):
            # use exiftool to save metadata to files
            with exiftool.ExifToolHelper() as et:
                tags = et.get_tags(
                    files=image_path,
                    tags=["XMP:RegionName", "XMP:RegionType", "XMP:PersonInImage"]
                )[0]
                if "XMP:PersonInImage" not in tags:
                    et.execute(f"-XMP:PersonInImage={face.person_shown}", image_path)
                elif face.person_shown not in tags["XMP:PersonInImage"]:
                    et.execute(f"-XMP:PersonInImage+={face.person_shown}", image_path)
                if "XMP:RegionName" not in tags:
                    execution_string = str("-XMP-mwg-rs:RegionInfo={AppliedToDimensions={"
                                           f"W={face.img_width}, H={face.img_height}, "
                                           "Unit=pixel}, RegionList=[{Area={"
                                           f"W={face.W}, H={face.H}, X={face.X}, Y={face.Y},"
                                           "Unit=normalized}, "
                                           f"Name={face.person_shown},"
                                           "Type=Face}]}")
                    # print(execution_string)
                    et.execute(execution_string, image_path)
                elif face.person_shown not in tags["XMP:RegionName"]:
                    execution_string = str("-XMP-mwg-rs:RegionList+=[{Area={"
                                           f"W={face.W}, H={face.H}, X={face.X}, Y={face.Y},"
                                           "Unit=normalized}, "
                                           f"Name={face.person_shown},"
                                           "Type=Face}]}")
                    # print(execution_string)
                    et.execute(execution_string, image_path)

    status_bar.empty()
