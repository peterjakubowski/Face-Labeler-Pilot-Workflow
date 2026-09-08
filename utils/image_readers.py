from pathlib import Path

import cv2
import numpy as np
import rawpy


def read_image_with_cv2(image_path: Path) -> np.array:
    """
    Returns a numpy array from an image path using opencv.
    :param image_path: Path to image
    :return: Image as numpy array
    """

    return cv2.imread(str(image_path))


def convert_thumbnail_to_array(thumbnail: rawpy.Thumbnail) -> np.array:
    """
    Returns numpy array from a rawpy.Thumbnail object.
    :param thumbnail: rawpy.Thumbnail object
    :return: Thumbnail image as a numpy array
    """

    thumbnail_image = np.frombuffer(thumbnail.data, dtype=np.uint8)

    return cv2.imdecode(thumbnail_image, cv2.IMREAD_COLOR)


def read_image_with_rawpy(image_path: Path) -> np.array:
    """
    Returns a numpy array from an image path using rawpy.
    :param image_path: Path to image
    :return: Image as numpy array
    """

    with rawpy.imread(str(image_path)) as raw_file:
        thumbnail = raw_file.extract_thumb()

    return convert_thumbnail_to_array(thumbnail)


def open_image(image_path: Path) -> np.array:
    """
    Returns a numpy array from an image path.
    :param image_path: Path to image
    :return: Image as numpy array
    """

    extension = image_path.suffix.strip(".").lower()

    if extension in ("jpg", "jpeg", "png", "tif", "tiff"):
        return read_image_with_cv2(image_path)
    elif extension in ("cr2", "dng", "nef"):
        return read_image_with_rawpy(image_path)

    return None