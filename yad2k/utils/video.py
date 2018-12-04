import glob
import imghdr
import os
import time

import cv2
import numpy as np


def rotate_image(image, angle):
    """
    Rotate image by arbitrary angle using OpenCV affine transforms.
    :param image: ndarray source image
    :param angle: float angle in degrees counterclockwise
    :return: ndarray rotated image
    """
    # TODO: fitting the whole image, without corner crop (as an option)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


class FPS:
    """Simple and naive frequency measurement."""

    def __init__(self, batch_size=1):
        self.start = None
        self.last = None
        self.current = None
        self.counter = 0
        self.batch_size = batch_size

    def tick(self):
        if self.start is None:
            self.start = time.time()
            self.last = self.start
            self.current = self.start
            return
        self.counter += 1  # we only increase it from sample 2 onwards
        self.last = self.current
        self.current = time.time()

    def report(self):
        if self.start is None or self.start == self.current:
            print('Not enough data to compute FPS.')
            return
        if self.batch_size > 1:
            print('effective FPS: avg={}, current={}'.format(
                self.counter / (self.current - self.start) * self.batch_size,
                1 / (self.current - self.last) * self.batch_size
            ))
        else:
            print('FPS: avg={}, current={}'.format(
                self.counter / (self.current - self.start),
                1 / (self.current - self.last)
            ))


class ImageDirectoryVideoCapture:
    """A mock of cv2.VideoCapture for reading image files from a directory."""

    def __init__(self, directory):
        self.directory = directory
        self.files = glob.iglob(os.path.join(self.directory, '*'))

    def read(self):
        try:
            while True:
                image_file = next(self.files)
                image_type = imghdr.what(image_file)
                if not image_type:
                    continue
                return True, cv2.imread(image_file)
        except StopIteration:
            return False, None
