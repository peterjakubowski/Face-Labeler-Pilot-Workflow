from pathlib import Path

import cv2
import numpy as np
import rawpy


def read_image_with_cv2(image_path: Path) -> np.ndarray:
    """
    Returns a numpy array from an image path using opencv.
    :param image_path: Path to image
    :return: Image as numpy array
    """

    return cv2.imread(str(image_path))


def convert_thumbnail_to_array(thumbnail: rawpy.Thumbnail) -> np.ndarray:
    """
    Returns numpy array from a rawpy.Thumbnail object.
    :param thumbnail: rawpy.Thumbnail object
    :return: Thumbnail image as a numpy array
    """

    thumbnail_image = np.frombuffer(thumbnail.data, dtype=np.uint8)

    return cv2.imdecode(thumbnail_image, cv2.IMREAD_COLOR)


def read_thumbnail_image_with_rawpy(image_path: Path) -> np.ndarray:
    """
    Returns a numpy array from an image path using rawpy by extracting its thumbnail.
    :param image_path: Path to image
    :return: Thumbnail image as a numpy array
    """

    with rawpy.imread(str(image_path)) as raw_file:
        thumbnail = raw_file.extract_thumb()

    return convert_thumbnail_to_array(thumbnail)


def read_raw_image_with_rawpy(image_path: Path) -> np.ndarray:
    """
    Returns a numpy array from an image path using rawpy to post process the raw image.
    :param image_path: Path to image
    :return: Post processed raw image as numpy array
    """

    with rawpy.imread(str(image_path)) as raw_file:
        raw_image = raw_file.postprocess()

    return raw_image


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
        thumbnail_image = read_thumbnail_image_with_rawpy(image_path)
        if thumbnail_image is not None and np.max(thumbnail_image.shape) > 1024:
            return thumbnail_image
        raw_image = read_raw_image_with_rawpy(image_path)
        return raw_image

    return None
