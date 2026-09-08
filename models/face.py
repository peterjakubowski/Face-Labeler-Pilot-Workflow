import cv2
import numpy as np

from utils.image_readers import open_image


class Face:
    """
    structure to store information about detected faces
    """

    def __init__(self,
                 img_path: str,
                 img_width: int,
                 img_height: int,
                 img_resized_width: int,
                 img_resized_height: int,
                 face_location: tuple[int, ...],
                 encoding: list) -> None:
        """
        Constructor for Face class objects.
        :param img_path: path to the image the face was detected in.
        :param img_width: original image width in pixels.
        :param img_height: original image height in pixels.
        :param img_resized_width: resized image width in pixels.
        :param img_resized_height: resized image height in pixels.
        :param face_location: location of the face in the image (top, right, bottom, left).
        :param encoding: encoding of the face detected in the image.
        """

        self.img_path = img_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_resized_width = img_resized_width
        self.img_resized_height = img_resized_height
        self.face_location = face_location
        self.encoding = encoding
        self.match_candidate = True
        self.person_shown = ""
        self.W = None
        self.H = None
        self.X = None
        self.Y = None
        self.normalize_face_location()

    def open_face_image(self) -> np.ndarray:
        """
        Opens the image containing the current face using cv2
        and crops the image to the region the face is in.
        :return: image (numpy.ndarray) cropped to the current face.
        """

        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _W, _H, _X, _Y = self.reverse_transform_face_location(width=img.shape[1], height=img.shape[0])
        img = img[_Y:_Y + _H, _X:_X + _W]
        return img

    def normalize_face_location(self) -> None:
        """
        Normalize/scale the face location coordinates.
        Bounding box (W, H, X, Y):
        Width of the bounding box.
        Height of the bounding box.
        X coordinate of the left of the bounding box.
        Y coordinate of the top of the bounding box.
        """

        top, right, bottom, left = self.face_location
        img_w, img_h = self.img_resized_width, self.img_resized_height
        self.W = round((right - left) / img_h, 4)
        self.H = round((bottom - top) / img_w, 4)
        self.X = round(left / img_h, 4)
        self.Y = round(top / img_w, 4)
        return

    def reverse_transform_face_location(self, width: int, height: int) -> tuple[int, ...]:
        """
        Reverse transform the normalized/scaled bounding box
        given the width and height of an image.
        :param width: width of image to scale bounding box to
        :param height: height of image to scale bounding box to
        :return: Reverse transformed bounding box (W, H, X, Y):
        Width of the bounding box.
        Height of the bounding box.
        X coordinate of the left of the bounding box.
        Y coordinate of the top of the bounding box.
        """

        if None in [self.W, self.H, self.X, self.Y]:
            self.normalize_face_location()
        _W = int(self.W * height)
        _H = int(self.H * width)
        _X = int(self.X * height)
        _Y = int(self.Y * width)
        return tuple([_W, _H, _X, _Y])
