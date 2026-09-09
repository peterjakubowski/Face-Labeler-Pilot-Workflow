# Face-Labeler-Pilot-Workflow

Face Labeler Pilot is an interactive human-in-the-loop Python-based workflow tool for photographers, digital asset managers, and anyone needing to tag people shown in photographs.

The tool is designed to assist in tagging photos from events, portrait sessions, or similar photo shoots that show the same people multiple times across many images. The tool performs best on images of people with their faces turned directly towards the camera, like in group shots and portraits.

The tool does not rely on any database of known faces or identities, rather it builds a list of known faces from the currrent session only and the names entered by the user. In its essence, a user need only enter the name of an individual once, thereafter, all representations of the individual is recognized if a match is found from previously labeled faces. This makes it a great tool for labeling faces that have never been seen before and may never be seen again by the photographer or editor. It only relies on the data from the current session and learns as it iterates through the session's images.

## Workflow Steps

1) Simply add a folder of images to the `watch_folder` directory at the root of the project and select it when prompted in the workflow. Supported file types are JPEG, PNG, and TIFF.

2) Click 'Detect Faces' to iterate over the images and let the face detection model get the all the face locations and face encodings.

3) After all the faces are found, the workflow will begin iterating over all the found faces and will prompt you to label them. Start by typing in the names for new unknown faces. Once a face has been labeled it will be added to a list of known faces. When there are known faces in the list, the face recognition algorithm will check for matches with the known faces and ask you to confirm its prediction if a match is found, otherwise you'll be prompted to enter a new name or choose a name from the list of names that have already been entered.

4) Click 'Write Metadata' to save/embed the face locations (bounding boxes) and names of the person(s) shown in the image's metadata. Face locations along with names are saved in the MGW Regions List uri. Names are also saved in the XMP:EXT4 PersonInImage field.

## Installation & Setup

Clone the repository

```commandline
git clone https://github.com/peterjakubowski/Face-Labeler-Pilot-Workflow.git
cd Face-Labeler-Pilot-Workflow
```

Before configuring your environment, you should have Python 3.10.12 installed. It's highly recommended that you create a virtual environment, either using Python's built-in virtual environment or Conda virtual environment.

### Python Virtual Environment

1. #### Create a virtual environment

    Navigate to the project directory and execute the built-in Python venv module:

    ```commandline
    python3 -m venv .venv
    ```

2. #### Activate the environment

    You must activate the environment to ensure your terminal uses this isolated instance of Python and pip

* #### macOS / Linux

    ```commandline
    source .venv/bin/activate
    ```

* #### Windows (Command Prompt)

    ```commandline
    .venv\Scripts\activate.bat
    ```

* #### Windows (PowerShell)

    ```commandline
    .venv\Scripts\Activate.ps1
    ```
    Once activated, your terminal prompt will show (.venv) at the beginning of the line.

3. #### Install the Requirements

    Make sure your terminal is in the same directory as your `requirements.txt` file, then run:

    ```commandline
    pip install -r requirements.txt
    ```

    The following dependencies will be installed as defined in the `requirements.txt` file:

    ```text
    numpy==1.26.4
    pandas==2.0.3
    streamlit==1.54.0
    opencv-python==4.9.0.80
    dlib==19.24.9
    face-recognition==1.3.0
    PyExifTool==0.5.6
    rawpy==0.27.1
   ```
   
### Conda Virtual Environment

Alternatively, create a new Conda virtual environment from the `environment.yml` file using the following command in your command prompt:

```
conda env update --file environment.yml --prune
```
The following dependencies will be installed as defined in the environment.yml file:

```
python 3.10.12
pandas 2.0.3
streamlit 1.54.0
opencv-python 4.9.0.80
numpy 1.26.4
dlib 19.24.9
face-recognition 1.3.0
PyExifTool 0.5.6
rawpy 0.27.1
```

### Requirements

| Dependency                                                                |          Category           |                                                                                                                                                                                                                                                       Primary Function & Usage in Repository |
|:--------------------------------------------------------------------------|:---------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [streamlit](https://docs.streamlit.io/)                                   |        UI Framework         |                                                                                                        Interactive Web Application: Serves as the web GUI framework in `app.py`. Powers the interactive dashboard, rendering image previews, labeling controls, and metadata export buttons. |
| [face-fecognition](https://pypi.org/project/face-recognition/)            |     Deep Learning / AI      |                                                                Face Detection & Embedding Extraction: Interface built on `dlib` (used in `utils/image_processing.py` and `app.py`) that locates faces in images and extracts 128-dimensional facial embedding vectors for identity matching. |
| [dlib](https://pypi.org/project/dlib/)                                    | Machine Learning Algorithms |                                                                                Underlying Vision Engine: Core C++ computer vision engine supporting face-recognition. Provides pre-trained facial landmark predictors and HOG/CNN face detectors for precise facial alignment and detection. |
| [opencv-python](https://pypi.org/project/opencv-python/)                  |      Image Processing       |                                                            Image Preprocessing & Manipulation: Utilized within `utils/image_processing.py` for reading images from disk, converting color spaces (BGR to RGB), cropping face bounding boxes $(x, y, w, h)$, and drawing visual box overlays. |
| [image_utils](https://github.com/peterjakubowski/Image-Editing-Utilities) | Image Processing Utilities  |                                                                                                                                        Image Path Helper: Used in the file workflow specifically to search, filter, and extract list sequences of image file paths across input directories. |
| [numpy](https://pypi.org/project/numpy/)                                  |    Scientific Computing     |                                                                                  Image Data Structure & Math: Essential matrix backend required by opencv-python (cv2), as images are loaded and manipulated directly as multi-dimensional NumPy arrays throughout the application pipeline. |
| [pandas](https://pypi.org/project/pandas/)                                |      Data Engineering       |                                                                                                                                                                  Interactive Table Display: Used within the Streamlit user interface to format, organize, and display structured DataFrames. |
| [PyExifTool](https://pypi.org/project/PyExifTool/)                        |          Metadata           | EXIF & XMP Metadata Writing: Python wrapper around `ExifTool` in `utils/exiftool.py` used to extract raw EXIF data and embed standardized Person Shown (PersonInImage) tags and MWG Regions (Metadata Working Group face bounding box regions and identity names) directly into image files. |
| [rawpy](https://github.com/letmaik/rawpy)                                 |    RAW Image Processing     |                                                                                                                                                                          RAW Image Processing: Python wrapper for `libraw` used to postprocess raw image files and extract their thumbnails. |

### Additional system requirements

#### Exiftool

Additionally, [Exiftool](https://exiftool.org/) must be installed on your system in order to read, write and edit image metadata using [PyExifTool](https://pypi.org/project/PyExifTool/). Installation instructions can be found on the Exiftool website [here](https://exiftool.org/install.html).

## Launching the tool

To launch the Streamlit server within the virtual environment, run the following command while the virtual environment is activated (in your command prompt, you should see `(.venv)` if using a Python virtual environment and `(facelabelerpilot_env)` if using conda):

```commandline
streamlit run app.py
```

Then open http://localhost:8501

## Future Improvements and Features

* Give the user an option to change the tolerance for finding a match.

* Add additional face detection algorithms (or replace the current with another) that can better detect faces turned to the side or in profile. The current algorithm in use does best at detected faces that are turned towards the camera.

* Try using an object detection model that first detects people (not faces) in an image before checking for faces  with the face detection algorithm. This could help increase the probability of finding a person in the image that needs to be tagged. A person without a face would always need to be manually tagged using this method since there would be no way to get a face encoding to compare to the list of known faces.

* Create some kind of option to add additional tags based on the person shown in the image. This would require the user to supply keys and values for data lookup to take place. A possible use case could be e-commerce on-figure photography where a particular model is associated with products and skus that must be tagged in a shot. Images would be placed in the watch folder organized by shots (shot number) in seperate subfolders. A csv file could be included to provide model names (key), shot number(key), and product/skus (values).
