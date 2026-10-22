import csv
from pathlib import Path

import streamlit as st

from config import IMG_DIR


def export_metadata_to_csv(select_folder: str):
    # open/create a csv file for writing
    csv_file_path: Path = IMG_DIR / select_folder / 'metadata.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        # initialize a csv writer object from the csv module
        csv_writer = csv.writer(csv_file, dialect='excel', delimiter=',')
        # define column names for the csv file
        csv_columns = ['filename', 'dim_w', 'dim_h', 'region_w', 'region_h', 'region_x', 'region_y', 'person_shown']
        # write column names to the first row of the csv file
        csv_writer.writerow(csv_columns)
        # iterate over the labeled data and write a row to the csv file for each face
        for fp, faces in st.session_state['labeled'].items():
            fp = Path(fp)
            for face in faces:
                csv_writer.writerow([
                    fp.name, face.img_width, face.img_height, face.W, face.H, face.X, face.Y, face.person_shown
                ])
