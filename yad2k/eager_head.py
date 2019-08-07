"""
The fully convolutional mode of YOLOv2 doesn't work... or does it?

This file is meant to solve this mystery, by writing the whole "yolo_head",
i.e. the conversion from model outputs (last layer filters) to:
- boxes
- objectness score
- class scores

A different flow:
decode a bit -> filter by score threshold -> compute box coordinates (should work for w != h) -> NMS.

"""

import attr
import numpy as np


@attr.s
class Anchor:
    x_idx = attr.ib(type=int)
    y_idx = attr.ib(type=int)
    score = attr.ib(type=float)
    multipliers = attr.ib()
    # this is too much information:
    raw_data = attr.ib(repr=False)
    box_center = attr.ib()
    box_size = attr.ib()
    class_idx = attr.ib()


def filter_anchors(features, anchors, threshold=0.5):
    batches = features.reshape(features.shape[:-1] + (len(anchors), -1))
    # num_classes = batches.shape[-1] - 5
    # print('filtering by t', threshold)

    results = []

    for batch in batches:
        anchors_above = []

        for y_idx, row in enumerate(batch):
            for x_idx, anchor_group in enumerate(row):
                for a_idx, record in enumerate(anchor_group):
                    score = sigmoid(record[4]) * np.max(softmax(record[5:]))

                    if score < threshold:
                        continue

                    #print('anchor', (x_idx, y_idx, a_idx), 'has score', score)
                    #import code; code.interact(local=locals())

                    anchors_above.append(Anchor(
                        x_idx,
                        y_idx,
                        np.array(score),  # not rounding, but making display acceptable
                        anchors[a_idx],  # anchor box multipliers
                        record,
                        sigmoid(record[:2]).astype('float64'),
                        np.exp(record[2:4]).astype('float64'),
                        np.argmax(record[5:])  # exp is a monotonous function
                    ))

        results.append(anchors_above)

    return results


def softmax(x):
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
