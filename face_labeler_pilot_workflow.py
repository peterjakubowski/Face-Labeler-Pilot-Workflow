import pandas as pd
import streamlit as st
from streamlit import session_state as sess
from streamlit_free_text_select import st_free_text_select
from imutils import paths
import face_recognition
import cv2
import exiftool
import os
import uuid
import time
from collections import defaultdict
from collections import deque
import textwrap


class Face:
    # structure to store information about detected faces
    def __init__(self):
        self.img_path = ""
        self.img_height = 0
        self.img_width = 0
        self.face_top = 0
        self.face_right = 0
        self.face_bottom = 0
        self.face_left = 0
        self.encoding = []
        self.match_candidate = True

    def face_location(self):
        return tuple([self.face_top, self.face_right, self.face_bottom, self.face_left])

    def current_face(self):
        # print(self.img_path, self.face_location(), self.encoding, self.match_candidate)
        return self.img_path, self.face_location(), self.encoding, self.match_candidate


# @ st.cache_data(show_spinner=False)
def strip_faces(img_paths):
    # initialize status bar
    _status_bar = st.progress(0, 'Firing up the face detection algorithm!')
    time.sleep(1)
    # keep a queue of found faces
    # queue is a list of tuples (image path, face bounding box, face encoding)
    q = deque()
    # iterate over all image paths is the selected directory and gather all detected faces and face encodings
    for i in range(len(img_paths)):
        # update progress
        _status_bar.progress((i + 1) / len(img_paths),
                             text=f'({i + 1} of {len(img_paths)}) Detecting faces in {img_paths[i].split("/")[-1]}...')
        # open image
        image = cv2.imread(img_paths[i])
        # convert image color from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize the image for fast inference
        resized_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        # detect face locations in image
        face_locations = face_recognition.face_locations(resized_image, model='hog')
        for face_location in face_locations:
            _top, _right, _bottom, _left = face_location
            # crop the image to the face
            # face_image = resized_image[top:bottom, left:right]
            # get face encoding
            encodings = face_recognition.face_encodings(resized_image,
                                                        known_face_locations=[face_location],
                                                        num_jitters=1,
                                                        model="large")
            face = Face()
            face.img_path = img_paths[i]
            face.face_top, face.face_right, face.face_bottom, face.face_left = [d * 4 for d in face_location]
            face.encoding = encodings

            # update the queue
            q.append(face)

    _status_bar.empty()

    return q


def record_name():
    if sess.selected_name:
        if sess.selected_name == 'Someone else':
            sess['faces_detected'][0].match_candidate = False
        else:
            if sess.selected_name == 'Not a face':
                sess.faces_count -= 1
            else:
                sess.name_options[sess.selected_name] += 1
                sess.face_i += 1
                if sess['faces_detected'][0].img_path not in sess.labeled:
                    sess.labeled[sess['faces_detected'][0].img_path] = {'face_locations': [], 'person_shown': []}
                sess.labeled[sess['faces_detected'][0].img_path]['face_locations'].append(sess['faces_detected'][0].face_location())
                sess.labeled[sess['faces_detected'][0].img_path]['person_shown'].append(sess.selected_name)
                if len(sess['faces_detected'][0].encoding) > 0:
                    sess.data['encodings'].append(sess['faces_detected'][0].encoding[0])
                    sess.data['names'].append(sess.selected_name)
            sess['faces_detected'].popleft()

    return


st.header("Face Labeler Pilot", divider='rainbow')
st.text(textwrap.dedent('''
1) Detect faces in images
2) Label detected faces
3) Embed names and face locations in the image's metadata
'''))

IMG_DIR = "watch_folder"

folder_names = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]

select_folder = st.selectbox(label='Choose a folder of images to scan for faces',
                             options=folder_names)

detect_faces = st.button(label="Detect Faces")

if detect_faces and select_folder:
    # list all the images in the selected folder
    image_paths = list(paths.list_images(os.path.join(IMG_DIR, select_folder)))
    # detect faces in all the images, get a list/queue of faces (instances of Face class)
    sess['faces_detected'] = strip_faces(image_paths)
    # count how many faces were detected
    sess['faces_count'] = len(sess.faces_detected)
    # count of faces labeled
    sess['face_i'] = 1
    # dictionary of labeled faces
    sess['labeled'] = {}
    # dictionary of face encodings and names
    sess['data'] = {'encodings': [], 'names': []}
    # dictionary of names/identities and counts
    sess['name_options'] = defaultdict(int)
    success_text = (f"Face detection is complete! "
                    f"Found {len(sess.faces_detected)} faces "
                    f"in {len(image_paths)} images."
                    )
    st.success(success_text, icon='✅')

if 'faces_detected' in sess:
    # check if there are faces in our queue
    if sess['faces_detected']:
        # ask the user if the workflow should automatically confirm/accept matches
        auto_confirm_matches = st.checkbox(label="Auto confirm matches?",
                                           value=False,
                                           key='auto_confirm_matches'
                                           )
        status_bar = st.progress(sess.face_i / sess.faces_count,
                                 text=f'Labeling face {sess.face_i} of {sess.faces_count}')
        # pop the next face from the queue
        current_face = sess['faces_detected'][0]
        # print(current_face.current_face())
        sess.img_path, sess.face_location, sess.encoding, sess.match = current_face.current_face()

        img = cv2.imread(sess.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_H, img_W, _ = img.shape
        top, right, bottom, left = sess.face_location
        img = img[top:bottom, left:right]
        # check if the current face has an encoding
        if len(sess.encoding) > 0:
            # compare the face encoding to existing encodings to see if we can find a match
            # note: the lower the tolerance, the more sensitive the algorithm is at matching faces
            matches = face_recognition.compare_faces(sess.data['encodings'],
                                                     sess.encoding[0],
                                                     tolerance=0.55)
            # st.write(matches)

            if True in matches:
                matched_indices = [i for (i, b) in enumerate(matches) if b]
                count = {}
                for i in matched_indices:
                    name = sess.data['names'][i]
                    count[name] = count.get(name, 0) + 1
                predicted_name = max(count, key=count.get)
                # st.write(list(sess.name_options.keys()))
                if not auto_confirm_matches:
                    with st.form(str(uuid.uuid4())):
                        st.image(img, width=100)
                        if sess.match:
                            st.write(f'I think this face belongs to **{predicted_name}**, can you confirm?')
                            selected_name = st.selectbox(label=('The predicted name has been pre-selected, '
                                                                'click the submit button to confirm.\n\n'
                                                                'Select "Not a face" to skip this face.\n\n'
                                                                'Select "Someone else" if their name '
                                                                'is not in the list.\n'
                                                                ),
                                                         options=['Not a face', 'Someone else'] + sorted(
                                                             sess.name_options.keys()),
                                                         index=sorted(sess.name_options.keys()).index(
                                                             predicted_name) + 2,
                                                         key='selected_name')
                        elif not sess.match:
                            st.write((f"I think this face belongs to **{predicted_name}**, "
                                      "but you think it's someone else, who do you think this is?"))
                            selected_name = st_free_text_select(label=('Who do you think this is? Type in a new name '
                                                                       'or select one from the list. '
                                                                       'Select "Not a face" to skip this face.'),
                                                                options=sorted(sess.name_options.keys()),
                                                                key="selected_name"
                                                                )
                        st.form_submit_button(label='Submit', on_click=record_name)
                elif auto_confirm_matches:
                    st.image(img, width=100)
                    st.write(f"This face belongs to **{predicted_name}**")
                    sess['selected_name'] = predicted_name
                    # wait for a moment, user can still interrupt by unchecking auto confirm matches
                    time.sleep(1)
                    record_name()
                    st.rerun()

            else:
                with st.form(str(uuid.uuid4())):
                    st.image(img, width=100)
                    st.write("I don't recognize this face, who is this?")
                    selected_name = st_free_text_select(label=('Type in a new name or select one from the list. '
                                                               'Select "Not a face" to skip this face.'),
                                                        options=['Not a face'] + sorted(
                                                            sess.name_options.keys()),
                                                        key="selected_name"
                                                        )
                    st.form_submit_button(label='Submit', on_click=record_name)
        else:
            with st.form(str(uuid.uuid4())):
                st.image(img, width=100)
                st.write('This face has no encoding. Is this a face?')
                selected_name = st_free_text_select(label=('Type in a new name or select one from the list. '
                                                           'Select "Not a face" to skip this face.'),
                                                    options=['Not a face'] + sorted(
                                                        sess.name_options.keys()),
                                                    key="selected_name"
                                                    )
                st.form_submit_button(label='Continue', on_click=record_name)
    # if our queue of faces is empty, check if we have labeled any images
    else:
        success_text = "All faces have been labeled!"
        st.success(success_text, icon='✅')
        df = pd.DataFrame(data=sess.name_options.items(),
                          columns=['names', 'counts'])
        df.set_index('names', inplace=True)
        st.dataframe(df.sort_index())
        # if we have labeled data, let's embed the face locations and names in the image metadata
        if 'labeled' in sess:
            st.write(sess.labeled)
            # st.write(sess.data)
            write_metadata = st.button(label="Write Metadata")
            if write_metadata:
                status_text = 'Begin writing metadata to files!'
                status_bar = st.progress(0, status_text)
                n = len(sess.labeled)
                j = 0
                for image_path, image_data in sess.labeled.items():
                    status_bar.progress((j + 1) / n,
                                        text=f'({j + 1} of {n}) Writing metadata to {image_path.split("/")[-1]}...')
                    img = cv2.imread(image_path)
                    img_h, img_w = img.shape[:2]
                    # with exiftool.ExifToolHelper() as et:
                    # et.execute("-XMP-mwg-rs:RegionInfo=", image_path)
                    # metadata = et.get_metadata(image_path)
                    for i, person_shown in enumerate(image_data['person_shown']):
                        top, right, bottom, left = image_data['face_locations'][i]
                        W = round((right - left) / img_h, 4)
                        H = round((bottom - top) / img_w, 4)
                        X = round(left / img_h, 4)
                        Y = round(top / img_w, 4)
                        with exiftool.ExifToolHelper() as et:
                            tags = et.get_tags(files=image_path,
                                               tags=["XMP:RegionName", "XMP:RegionType", "XMP:PersonInImage"])[0]
                            # st.write(tags)
                            if "XMP:PersonInImage" not in tags:
                                # st.write(f'Setting PersonInImage to {person_shown}')
                                et.execute(f"-XMP:PersonInImage={person_shown}", image_path)
                            elif person_shown not in tags["XMP:PersonInImage"]:
                                # st.write(f'Appending {person_shown} to PersonInImage')
                                et.execute(f"-XMP:PersonInImage+={person_shown}", image_path)

                            if "XMP:RegionName" not in tags:
                                # st.write(f'Setting RegionName to {person_shown}')
                                execution_string = str("-XMP-mwg-rs:RegionInfo={AppliedToDimensions={"
                                                       f"W={img_w}, H={img_h}, "
                                                       "Unit=pixel}, RegionList=[{Area={"
                                                       f"W={W}, H={H}, X={X}, Y={Y},"
                                                       "Unit=normalized}, "
                                                       f"Name={person_shown},"
                                                       "Type=Face}]}")
                                print(execution_string)
                                et.execute(execution_string, image_path)
                            elif person_shown not in tags["XMP:RegionName"]:
                                # st.write(f'Appending {person_shown} to RegionName')
                                execution_string = str("-XMP-mwg-rs:RegionList+=[{Area={"
                                                       f"W={W}, H={H}, X={X}, Y={Y},"
                                                       "Unit=normalized}, "
                                                       f"Name={person_shown},"
                                                       "Type=Face}]}")
                                print(execution_string)
                                et.execute(execution_string, image_path)
                    j += 1
                status_bar.empty()
                st.success("Metadata saved to files! Workflow complete!", icon='✅')
