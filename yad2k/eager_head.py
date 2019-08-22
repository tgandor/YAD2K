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
    x_idx = attr.ib(type=int, repr=False, default=0)
    y_idx = attr.ib(type=int, repr=False, default=0)
    score = attr.ib(type=float, default=0)
    multipliers = attr.ib(repr=False, default=None)  # anchor relative factors
    raw_data = attr.ib(repr=False, default=None)  # this is too much information, so don't repr it!
    box_center = attr.ib(default=[0, 0])
    box_size = attr.ib(repr=True, default=[1, 1])
    class_idx = attr.ib(type=int, default=None)

    def corners(self, image_width=1, image_height=1):
        image_size = np.array([image_width, image_height])
        half_size = self.box_size * image_size / 2
        abs_center = self.box_center * image_size
        return np.array([abs_center - half_size, abs_center + half_size]).round(0).astype(np.int)

    @property
    def area(self):
        return self.box_size[0] * self.box_size[1]

def filter_anchors(features, anchors, threshold=0.5):
    batches = features.reshape(features.shape[:-1] + (len(anchors), -1))
    # num_classes = batches.shape[-1] - 5
    # print('filtering by t', threshold)

    results = []

    for batch in batches:
        anchors_above = []
        num_cells_y = len(batch)
        num_cells_x = len(batch[0])
        grid_size = np.array([num_cells_x, num_cells_y])

        for y_idx, row in enumerate(batch):
            for x_idx, anchor_group in enumerate(row):
                grid_position = np.array([x_idx, y_idx])

                for a_idx, record in enumerate(anchor_group):
                    score = sigmoid(record[4]) * np.max(softmax(record[5:]))

                    if score < threshold:
                        continue

                    #print('anchor', (x_idx, y_idx, a_idx), 'has score', score)
                    #import code; code.interact(local=locals())

                    box_center = (sigmoid(record[:2]) + grid_position) / grid_size
                    box_size = np.exp(record[2:4]) * anchors[a_idx] / grid_size

                    anchors_above.append(Anchor(
                        x_idx,
                        y_idx,
                        np.array(score),  # not rounding, but making display acceptable
                        anchors[a_idx],  # anchor box multipliers
                        record,
                        box_center,
                        box_size,
                        np.argmax(record[5:]),  # exp is a monotonous function
                    ))

        results.append(anchors_above)

    return results


def interval_intersection(x_1, w_1, x_2, w_2):
    if abs(x_2 - x_1) > (w_1 + w_2) / 2:
        return 0

    points = sorted([x_1 - w_1/2, x_1 + w_1 / 2, x_2 - w_2 / 2, x_2 + w_2 / 2])
    return points[2] - points[1]


def intersection_over_union(a_1 : Anchor, a_2 : Anchor):
    """Calculate IoU - a.k.a. Jaccard index of 2 boxes.

    >>> intersection_over_union(Anchor(box_center=[0, 0], box_size=[2, 2]), Anchor(box_center=[0, 0], box_size=[4, 4]))
    0.25
    """
    x_overlap = interval_intersection(a_1.box_center[0], a_1.box_size[0], a_2.box_center[0], a_2.box_size[0])
    y_overlap = interval_intersection(a_1.box_center[1], a_1.box_size[1], a_2.box_center[1], a_2.box_size[1])
    intersection = x_overlap * y_overlap
    union = a_1.area + a_2.area - intersection
    return intersection / union


def non_maximum_suppression(anchors, iou_threshold=0.5):
    """Eliminate overlapping boxes with lower scores.

    I know, quadratic. Need to read some Adrian from pyimagesearch, maybe he knows
    how to do it in N log N."""
    results = []

    for anchor in sorted(anchors, key=lambda x: x.score, reverse=True):
        if any(intersection_over_union(anchor, accepted) > iou_threshold for accepted in results):
            continue

        results.append(anchor)

    return results


def softmax(x):
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
