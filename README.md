# Face-Labeler-Pilot-Workflow

Face Labeler Pilot is a Python-based workflow tool for photographers, digital asset managers, and anyone needing to tag people shown in photographs.

The tool is designed to assist in tagging photos from events, portrait sessions, or similar photo shoots that show the same people multiple times across many images. The tool performs best on images of people with their faces turned directly towards the camera, like in group shots and portraits.

The tool does not rely on any database of known faces or identities, rather it builds a list of known faces from the currrent session only and the names entered by the user. In its essence, a user need only enter the name of an individual once, thereafter, all representations of the individual is recognized if a match is found from previously labeled faces. This makes it a great tool for labeling faces that have never been seen before and may never be seen again by the photographer or editor. It only relies on the data from the current session and learns as it iterates through the session's images.

## Usage

1) Simply add a folder of images to the `watch_folder` directory at the root of the project and select it when prompted in the workflow. Supported file types are JPEG, PNG, and TIFF.

2) Click 'Detect Faces' to iterate over the images and let the face detection model get the all the face locations and face encodings.

3) After all the faces are found, the workflow will begin iterating over all the found faces and will prompt you to label them. Start by typing in the names for new unknown faces. Once a face has been labeled it will be added to a list of known faces. When there are known faces in the list, the face recognition algorithm will check for matches with the known faces and ask you to confirm its prediction if a match is found, otherwise you'll be prompted to enter a new name or choose a name from the list of names that have already been entered.

4) Click 'Write Metadata' to save/embed the face locations (bounding boxes) and names of the person(s) shown in the image's metadata. Face locations along with names are saved in the MGW Regions List uri. Names are also saved in the XMP:EXT4 PersonInImage field.

## Environment Setup and Dependencies


Create a new Conda virtual environment from the environment.yml file using the following command in your command prompt:

```
conda env update --file environment.yml --prune
```
The following dependencies will be installed as defined in the environment.yml file:

```
python 3.10.12
pandas 2.0.3
streamlit 1.32.2
streamlit-free-text-select 0.0.5
opencv-python 4.9.0.80
imutils 0.5.4
numpy 1.25.2
dlib 19.23.1
face-recognition 1.3.0
PyExifTool 0.5.6
```

Additionally, [Exiftool](https://exiftool.org/) must be installed on your system in order to read, write and edit metadata using [PyExifTool](https://pypi.org/project/PyExifTool/). Installation instructions can be found on the Exiftool website [here](https://exiftool.org/install.html).

## Launching the tool

To launch the streamlit server within the virtual environment, run the following command while the virtual environment is activated (you should see (facelabelerpilot_env) in your command prompt):

```
streamlit run face_labeler_pilot_workflow.py
```

Then open http://localhost:8501

## Future Improvements and Features

* Add an option to let the tool approve matches without asking the user for confirmation based on some criteria like the number of matches or how close in distance a match is. The tool currently asks the user to approve the name for every face encountered.

* Give the user an option to change the tolerance for finding a match.

* Add additional face detection algorithms (or replace the current with another) that can better detect faces turned to the side or in profile. The current algorithm in use does best at detected faces that are turned towards the camera.

* Try using an object detection model that first detects people (not faces) in an image before checking for faces  with the face detection algorithm. This could help increase the probability of finding a person in the image that needs to be tagged. A person without a face would always need to be manually tagged using this method since there would be no way to get a face encoding to compare to the list of known faces.

* Create some kind of option to add additional tags based on the person shown in the image. This would require the user to supply keys and values for data lookup to take place. A possible use case could be e-commerce on-figure photography where a particular model is associated with products and skus that must be tagged in a shot. Images would be placed in the watch folder organized by shots (shot number) in seperate subfolders. A csv file could be included to provide model names (key), shot number(key), and product/skus (values).

* Add a button to export the data to a csv file. Include the filename or image path, names, and face locations.

