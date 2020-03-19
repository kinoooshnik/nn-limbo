import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    sam_num = prediction.shape[0]
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for i in range(sam_num):
        if ground_truth[i] and prediction[i]:
            true_pos += 1
        elif ground_truth[i] and not prediction[i]:
            false_neg += 1
        elif not ground_truth[i] and prediction[i]:
            false_pos += 1
        elif not ground_truth[i] and not prediction[i]:
            true_neg += 1

    precision = 0
    recall = 0
    f1 = 0
    accuracy = 0

    if (true_pos + false_pos) != 0:
        precision = true_pos / (true_pos + false_pos)
    if (true_pos + false_neg) != 0:
        recall = true_pos / (true_pos + false_neg)
    if (true_pos + false_neg + true_neg + false_pos) != 0:
        accuracy = (true_pos + true_neg) / (true_pos + false_neg + true_neg + false_pos)
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    tp = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            tp += 1

        if prediction.shape[0] != 0:
            accuracy = tp / prediction.shape[0]
        else:
            accuracy = 0
    return accuracy
