import os
import re

from keras.models import load_model as load_keras_model
import numpy as np


def load_model(model_path, classes_path=None, anchors_path=None):
    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

    if not os.path.isfile(model_path):
        raise ValueError('Model file: {} - not found'.format(model_path))

    h5_suffix = re.compile(r'\.h5$')

    if anchors_path is None:
        anchors_path = h5_suffix.sub('_anchors.txt', model_path)
    else:
        anchors_path = os.path.expanduser(anchors_path)

    if not os.path.isfile(anchors_path):
        raise ValueError('Anchors file: {} - not found'.format(anchors_path))

    if classes_path is None:
        classes_path = h5_suffix.sub('_classes.txt', model_path)
    else:
        classes_path = os.path.expanduser(classes_path)

    if not os.path.isfile(classes_path):
        raise ValueError('Classes file: {} - not found'.format(classes_path))

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_keras_model(model_path)
    # yolo_model.compile('adam')

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), (
        'Output channels ({}) != anchors ({}) * {} ({} classes + 5 params)'.format(
            model_output_channels, num_anchors, num_classes + 5, num_classes
        )
    )

    return yolo_model, class_names, anchors
