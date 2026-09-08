import csv
import os

import streamlit as st

from config import IMG_DIR


def export_metadata_to_csv(select_folder: str):
    # open/create a csv file for writing
    with open(os.path.join(IMG_DIR, select_folder, 'metadata.csv'), 'w', newline='') as csv_file:
        # initialize a csv writer object from the csv module
        csv_writer = csv.writer(csv_file, dialect='excel', delimiter=',')
        # define column names for the csv file
        csv_columns = ['filename', 'region_w', 'region_h', 'region_x', 'region_y', 'person_shown']
        # write column names to the first row of the csv file
        csv_writer.writerow(csv_columns)
        # iterate over the labeled data and write a row to the csv file for each face
        for fp, faces in st.session_state['labeled'].items():
            filename = fp.split('/')[-1]
            for face in faces:
                csv_writer.writerow([filename, face.W, face.H, face.X, face.Y, face.person_shown])
