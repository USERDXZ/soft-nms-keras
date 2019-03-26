# -*- coding:utf-8 -*-
# Author:GuGu
"""
This is a keras version used to implement the Soft NMS algorithm.
Original Paper：Improving Object Detection With One Line of Code
"""
import tensorflow as tf
from keras import backend as K

def softnms(boxes, scores, thresh=0.001, sigma=0.5):
    """
    :param boxes:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param scores: 每个 boxes 对应的分数
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = boxes.get_shape()[0]
    indexes = K.reshape(K.cast(K.arange(N), dtype='float32'), (N,1))
    boxes = K.concatenate((boxes, indexes), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        tscore = K.gather(scores, i)
        pos = i + 1

        if i != N - 1:
            maxscore = K.max(scores[pos:], axis=0)
            maxpos = K.argmax(scores[pos:], axis=0)

            boxes = tf.cond(tscore < maxscore,
                           lambda: K.concatenate([boxes[:i, :], [boxes[maxpos + i + 1, :]],
                                                  boxes[i + 1:maxpos + i + 1, :], [boxes[i, :]],
                                                  boxes[maxpos + i + 2:, :]], axis=0),
                           lambda: boxes)
            scores = tf.cond(tscore < maxscore,
                             lambda: K.concatenate([scores[:i], [scores[maxpos + i + 1]],
                                                    scores[i + 1:maxpos + i + 1], [scores[i]],
                                                    scores[maxpos + i + 2:]],axis=0),
                             lambda: scores)
            areas = tf.cond(tscore < maxscore,
                            lambda: K.concatenate([areas[:i], [areas[maxpos + i + 1]],
                                                   areas[i + 1:maxpos + i + 1], [areas[i]],
                                                   areas[maxpos + i + 2:]],axis=0),
                            lambda: areas)
        # IoU calculate
        xx1 = K.maximum(boxes[i, 1], boxes[pos:, 1])
        yy1 = K.maximum(boxes[i, 0], boxes[pos:, 0])
        xx2 = K.minimum(boxes[i, 3], boxes[pos:, 3])
        yy2 = K.minimum(boxes[i, 2], boxes[pos:, 2])

        w = K.maximum(0.0, xx2 - xx1 + 1)
        h = K.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        weight = K.exp(-(ovr * ovr) / sigma)# gaussian
        scores = K.concatenate([scores[:pos], weight * scores[pos:]], axis=0)

    # select the boxes and keep the corresponding indexes
    inds = tf.boolean_mask(boxes[:, 4], scores > thresh)
    keep = K.cast(inds, dtype='int32')

    return keep