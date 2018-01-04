#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head

import cv2
import time

from itertools import count

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images.')
parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
         'of a YOLO_v2 model',
    nargs='?',
    default='model_data/yolo.h5')
parser.add_argument(
    '--source', '-i',
    help='video source (file path) for OpenCV VideoCapture, default: webcam 0', default=0)
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)
parser.add_argument(
    '-t',
    '--test',
    action='store_true',
    help='suppress display, boxes painting: show box count and FPS')
parser.add_argument(
    '--no-label',
    action='store_true',
    help='supress drawing class and confidence labels next to boxes')
parser.add_argument(
    '--max-boxes',
    type=int,
    default=10,
    help='max_boxes returned from yolo_eval'
)
parser.add_argument(
    '--box-thickness',
    type=int,
    default=0,
    help='thickness of drawn bounding boxes, in pixels'
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    help='number of images in a single batch'
)
parser.add_argument(
    '--fake-batch',
    action='store_true',
    help='repeat a single input frame --batch-size times (for testing performance)'
)
parser.add_argument(
    '--rotation',
    type=float,
    default=0.,
    help='rotation of images before processing in degrees counterclockwise'
)
parser.add_argument(
    '--limit', '-n',
    type=int,
    default=0,
    help='max number frames to process, 0 means no limit'
)


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


def _main(args):
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        max_boxes=args.max_boxes,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    cap = cv2.VideoCapture(args.source)
    fps = FPS(args.batch_size)
    frame_idx_generator = range(1, args.limit+1) if args.limit > 0 else count(1)
    batch = []

    for frame_idx in frame_idx_generator:
        ret, cv_image = cap.read()
        if args.rotation != 0.:
            cv_image = rotate_image(cv_image, args.rotation)

        if not ret or cv_image is None:
            print('Error grabbing:', ret, cv_image)
            break

        in_h, in_w = cv_image.shape[:2]

        if is_fixed_size:
            cv_image = cv2.resize(cv_image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (in_w - (in_w % 32),
                              in_h - (in_h % 32))
            cv_image = cv2.resize(cv_image, new_image_size)

        h, w = cv_image.shape[:2]

        image_data = cv_image.astype(np.float)
        image_data /= 255.

        if args.batch_size == 1:
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        elif not args.fake_batch:
            batch.append(image_data)
            if frame_idx % args.batch_size != 0:
                continue
            # print('process batch')
            image_data = np.stack(batch)
            batch = []
        elif args.fake_batch:
            # take batch
            image_data = np.stack([image_data] * args.batch_size)

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [w, h],
                K.learning_phase(): 0
            })

        fps.tick()
        fps.report()

        if args.test:
            continue

        print('Found {} boxes for frame {}'.format(len(out_boxes), frame_idx))

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * h + 0.5).astype('int32'))

        thickness = args.box_thickness or (w + h) // 300

        image = Image.fromarray(cv_image)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
            right = min(w, np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            if not args.no_label:
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        cv_image2 = cv2.resize(np.array(image), (in_w, in_h))
        cv2.imshow('Yolo2', cv_image2)
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
    sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())
