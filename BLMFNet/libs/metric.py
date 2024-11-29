import numpy as np


def confusion_matrix(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(num_classes * label[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf_mat


# def evaluate(conf_mat):
#     acc = np.diag(conf_mat).sum() / conf_mat.sum()
#     acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
#     acc_cls = np.nanmean(acc_per_class)
#     IoU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
#     mean_IoU = np.nanmean(IoU)
#     # 求kappa
#     pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum()**2)
#     kappa = (acc - pe) / (1 - pe)
#     return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa
def evaluate(conf_mat):
    # soomth 加个平滑项 免得除0
    smooth = 1e-5
    # 计算 Precision、Recall 和 F1 Score
    true_positive = np.diag(conf_mat)[1]
    false_positive = np.sum(conf_mat, axis=0)[1] - true_positive
    false_negative = np.sum(conf_mat, axis=1)[1] - true_positive

    precision = true_positive / (true_positive + false_positive + smooth)
    recall = true_positive / (true_positive + false_negative + smooth)
    f1_score = 2 * (precision * recall) / (precision + recall + smooth)

    # 计算 IOU
    iou = true_positive / (true_positive + false_positive + false_negative + smooth)

    return precision, recall, f1_score, iou