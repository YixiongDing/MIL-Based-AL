"""
This code is implemented as a part of the following paper and it is only meant to reproduce the results of the paper:
    "Active Learning for Deep Detection Neural Networks,
    "Hamed H. Aghdam, Abel Gonzalez-Garcia, Joost van de Weijer, Antonio M. Lopez", ICCV 2019
_____________________________________________________

Developer/Maintainer:  Hamed H. Aghdam
Year:                  2018-2019
License:               BSD
_____________________________________________________

"""

import numpy as np
__use_numpy__ = True


def boundingbox_iou(bb1, bb2):
    """
    计算两个bbox的intersection over union
    computes IoU of two bounding boxes
    :param bb1: [x1, y1, x2, y2]
    :param bb2: [x1. y1. x2. y2]
    :return:
    """
    x1 = max([bb1[0], bb2[0]])
    y1 = max([bb1[1], bb2[1]])
    x2 = min([bb1[2], bb2[2]])
    y2 = min([bb1[3], bb2[3]])
    if x1 > x2 or y1 > y2:
        intersection = 0
    else:
        intersection = float((x2 - x1) * (y2 - y1))

    iou = intersection / ((bb1[2] - bb1[0]) * (bb1[3] - bb1[1]) + (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]) - intersection)
    iou = 0 if iou < 0 else iou

    assert 0 <= iou <= 1
    return iou


def boundingbox_iou_matrix(list_bb_actual, list_bb_predicted):
    """
    计算实际bbox和预测bbox的iou，输出一个矩阵
    """
    m = len(list_bb_actual)
    n = len(list_bb_predicted)

    iou_mat = list()
    for i in xrange(m):
        cols = list()
        for j in xrange(n):
            iou = boundingbox_iou(list_bb_actual[i], list_bb_predicted[j])

            cols.append(iou)
        iou_mat.append(cols)

    if __use_numpy__:
        iou_mat = np.asarray(iou_mat, dtype=np.float32)

    return iou_mat


def iou_matrix_to_confusion_matrix(iou_matrix, min_iou_threshold):
    """
    只用于二分类问题
    Applicable only on **binary classification** problem
    :param iou_matrix: #GT x #PRED matrix
    :param min_iou_threshold:
    :return:
    """
    if __use_numpy__:
        iou_matrix_binary = np.where(np.less_equal(iou_matrix, min_iou_threshold), 0, 1)
        if iou_matrix.ndim != 2:
            fp = 0
            fn = iou_matrix.shape[0]
            tp = 0
        else:
            fp = iou_matrix_binary.shape[1] - np.count_nonzero(np.sum(iou_matrix_binary, axis=0, keepdims=True))
            fn = iou_matrix_binary.shape[0] - np.count_nonzero(np.sum(iou_matrix_binary, axis=1, keepdims=True))
            tp = np.count_nonzero(np.sum(iou_matrix_binary, axis=1, keepdims=True))

        conf_mat = np.asarray([[tp, fn], [fp, -1]], dtype=np.float32)
    else:
        raise NotImplementedError()

    return conf_mat


def compute_pre_recal(cm, mode='global'):
    """
    用两种方法计算precision和recall。计算FP, TP, FN在：1，整个数据集上；2，每张图片上
    Computes the precision and recall using two different methods. In the first method, it counts FP, TP and FN
    over the entire dataset. In the second method, it computes FP, FN and TP for each image separately.

    While both approaches are valid, each of them evaluates the method from different perspectives.
    """
    cm = cm.astype('float32')
    if mode == 'global':
        tp = np.sum(cm, axis=2)[0, 0]
        fp = np.sum(cm, axis=2)[1, 0]
        fn = np.sum(cm, axis=2)[0, 1]
        pre = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
    elif mode == 'average':
        # ##################################################################################################
        # This approach is less sensitive to outliers.
        # ##################################################################################################
        tp = cm[0, 0]
        fp = cm[1, 0]
        fn = cm[0, 1]
        pre = np.mean(tp / (tp + fp + 1e-9))
        rec = np.mean(tp / (tp + fn + 1e-9))
    else:
        raise ValueError("mode can only take on of the following values: 'global', 'local'")

    return pre, rec


def compute_miss_rate_fppi(cm, mode='average'):
    """
    用来计算MR-FPPI（Miss rate against false positives per image）是object detection中评估性能一个指标
    """
    pre, rec = compute_pre_recal(cm, mode)
    fp = cm[1, 0]
    fppi = np.mean(fp)
    miss = 1 - rec
    return miss, fppi, pre, rec
