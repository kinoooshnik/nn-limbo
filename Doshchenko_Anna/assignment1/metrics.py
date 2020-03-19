def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    length = ground_truth.shape[0]
    for i in range(length):

        if prediction[i] == 0:
            if ground_truth[i] == 0:
                TN += 1
            else:
                FN += 1
        else:
            if ground_truth[i] == 1:
                TP += 1
            else:
                FP += 1
    accuracy = (TP + TN) / length
    if TP + FP != 0:
        precision = TP / (TP + FP)
    if TP + FN != 0:
        recall = TP / (TP + FN)
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
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
    length = ground_truth.shape[0]
    true, accuracy = 0, 0
    for i in range(length):
        if prediction[i] == ground_truth[i]:
            true += 1
    if length != 0:
        accuracy = true/length
    return accuracy
