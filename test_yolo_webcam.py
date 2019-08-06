#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import datetime
import os
import random
import re
import time

from itertools import count

parser = argparse.ArgumentParser(description='Run a YOLO_v2 style detection model on test images.')
parser.add_argument('model_path', help='h5 file containing a YOLO_v2 model', default='model_data/yolo.h5')
parser.add_argument('--source', '-i', help='video file path for OpenCV VideoCapture', default=0)
parser.add_argument('-a', '--anchors_path', help='path to anchors file')
parser.add_argument('-c', '--classes_path', help='path to classes file')
parser.add_argument('-o', '--output_path', help='path to test output, defaults to images/out')
parser.add_argument('-s', '--score_threshold', type=float, help='min objectness score [=.3]', default=.3)
parser.add_argument('-iou', '--iou_threshold', type=float, help='max IOU for non max suppression [=.5]', default=.5)
parser.add_argument('-t', '--test', action='store_true', help='suppress display, boxes painting: show box count and FPS')
parser.add_argument('--nobgr', action='store_true', help='turn off BGR->RGB conversion for images')
parser.add_argument('--no-label', action='store_true', help='draw boxes without class labels')
parser.add_argument('--original', action='store_true', help='show original image also')
parser.add_argument('--max-boxes', type=int, default=50, help='max_boxes returned from yolo_eval')
parser.add_argument('--box-thickness', type=int, default=1, help='thickness of drawn bounding boxes, in pixels')
parser.add_argument('--batch-size', type=int, default=1, help='number of images in a single batch')
parser.add_argument('--fake-batch', action='store_true', help='repeat a single input frame --batch-size times (for testing performance)')
parser.add_argument('--rotation', type=float, default=0., help='rotation of images before processing in degrees counterclockwise')
parser.add_argument('--limit', '-n', type=int, default=0, help='max number frames to process, 0 means no limit')
parser.add_argument('--downsample', type=int, default=0, help='downsample input image N times (row and column stride)')
parser.add_argument('--square-crop', type=int, default=0, help='cut out a NxN square of the image (for fixed size testing)')
parser.add_argument('--skip-frames', type=int, default=0, help='number of frames to skip at the beginning of capture')
parser.add_argument('--dump', action='store_true', help='save image and processed result')
parser.add_argument('--wait', '-w', type=int, default=1, help='argument for cv2.waitKey()')
parser.add_argument('--dump-all', action='store_true', help='save all images even if no objects')


def _main(args):
    start = datetime.datetime.now()
    wall = lambda: datetime.datetime.now() - start

    import cv2
    from yad2k.utils.video import rotate_image, FPS, ImageDirectoryVideoCapture

    if os.path.isdir(str(args.source)):
        cap = ImageDirectoryVideoCapture(args.source)
    elif re.match(r'\d+$', str(args.source)):
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    print(wall(), 'Capture', args.source, 'opened. Loading Keras...')

    import colorsys
    import numpy as np
    from keras import backend as K
    from yad2k.models import utils
    from PIL import Image, ImageDraw, ImageFont

    from yad2k.models.keras_yolo import yolo_eval, yolo_head

    if args.skip_frames > 0:
        for i in range(args.skip_frames):
            status, frame = cap.read()
            if not status:
                cap.close()
                print(wall(), 'Failed acquisition while skipping initial frames, on frame', i)
                exit()
        print(wall(), 'Skipped', args.skip_frames, 'frames')

    yolo_model, class_names, anchors = utils.load_model(args.model_path, args.classes_path, args.anchors_path)
    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    print(wall(), 'Model image size:', model_image_size)
    is_fixed_size = model_image_size != (None, None)
    print(wall(), 'Fixed size model found.' if is_fixed_size else 'Variable, quantized size model.')
    print(wall(), 'Anchors:', anchors)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if not args.test:
        font = None

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

    fps = FPS(args.batch_size)
    frame_idx_generator = range(1, args.limit + 1) if args.limit > 0 else count(1)
    batch = []

    print(wall(), 'Entering main loop.')

    for frame_idx in frame_idx_generator:
        ret, cv_image = cap.read()

        if not ret or cv_image is None:
            print(wall(), 'Error grabbing:', ret, cv_image)
            break

        # preprocessing
        if args.downsample > 1:
            cv_image = cv_image[::args.downsample, ::args.downsample]

        if args.square_crop:
            cv_image = cv_image[:args.square_crop, :args.square_crop]

        if args.rotation != 0.:
            cv_image = rotate_image(cv_image, args.rotation)

        # resize to YOLO input shape
        in_h, in_w = cv_image.shape[:2]

        original = cv_image

        if is_fixed_size:
            cv_image = cv2.resize(cv_image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (in_w - (in_w % 32),
                              in_h - (in_h % 32))
            cv_image = cv2.resize(cv_image, new_image_size)

        h, w = cv_image.shape[:2]

        image_data = cv_image
        # thanks, Mateusz! Heard about it so many times, and still fell for it!
        if not args.nobgr:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB).astype(np.float)

        image_data = image_data.astype(np.float)
        image_data /= 255.

        # prepare input batch
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

        # actual yolo processing
        out_boxes, out_scores, out_classes, raw_output = sess.run(
            [boxes, scores, classes, yolo_model.output],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [w, h],
                K.learning_phase(): 0
            })

        print('Raw', raw_output.shape, 'first cell:', raw_output[0, 0, 0].reshape(len(anchors), -1))

        # measure performance
        fps.tick()
        fps.report()

        if args.test:
            continue

        # report results
        print(wall(), 'Found {} boxes for frame {}'.format(len(out_boxes), frame_idx))

        if font is None:
            font = ImageFont.truetype(
                font=os.path.join(os.path.dirname(__file__), 'font', 'FiraMono-Medium.otf'),
                size=np.floor(3e-2 * h + 0.5).astype('int32'))

        thickness = args.box_thickness or (w + h) // 300

        # image = Image.fromarray(original)
        # we should stick to relative coordinates 0..1
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
        if args.original:
            cv2.imshow('Original', original)
        if args.dump and (len(out_boxes) or args.dump_all):
            cv2.imwrite('frame_{:.2f}.jpg'.format(time.time()), original)
            cv2.imwrite('yolo_{:.2f}.jpg'.format(time.time()), cv_image2)
        key = cv2.waitKey(args.wait)
        if key & 0xff == ord('q'):
            break
    sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())
