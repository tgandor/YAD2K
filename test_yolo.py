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
from yad2k.models.utils import load_model

parser = argparse.ArgumentParser(description='Run a YOLO_v2 detection model on test images')
parser.add_argument('model_path', help='h5 file containing a YOLO_v2 model')
parser.add_argument('-a', '--anchors_path', help='path to anchors file')
parser.add_argument('-c', '--classes_path', help='path to classes file')
parser.add_argument('-t', '--test_path', help='path to directory of test images, defaults to images/', default='images')
parser.add_argument('-o', '--output_path', help='path to test output, defaults to images/out', default='images/out')
parser.add_argument('-s', '--score_threshold', type=float, help='min objectness score [=.3]', default=.3)
parser.add_argument('-iou', '--iou_threshold', type=float, help='max IOU for non max suppression [=.5]', default=.5)
parser.add_argument('--gpu', help='specify GPU to use (0, 1, etc.)')


def _main(args):
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.makedirs(output_path)

    if args.gpu:
        # https://github.com/keras-team/keras/issues/3685
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo_model, class_names, anchors = load_model(args.model_path, args.classes_path, args.anchors_path)
    print('{} model, anchors, and classes loaded.'.format(args.model_path))

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

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
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        max_boxes=20,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    for image_file in sorted(os.listdir(test_path)):
        try:
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                continue
        except (IsADirectoryError, PermissionError):
            continue

        image = Image.open(os.path.join(test_path, image_file))

        if image.mode != 'RGB':
            image = image.convert(mode='RGB')

        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

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
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
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
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image.save(os.path.join(output_path, image_file), quality=90)
    sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())
