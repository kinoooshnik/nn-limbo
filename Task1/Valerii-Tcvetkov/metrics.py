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
    f1 = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            if prediction[i] == True:
                tp += 1
            else:
                tn += 1
        else:
            if prediction[i] == True:
                fp += 1
            else:
                fn += 1

    # print(prediction)
    # print(ground_truth)
    # print(tp + tn + fp + fn)

    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if (tp + tn + fp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy = 0

    if (2 * tp + fp + fn) != 0:
        f1 = (2 * tp) / (2 * tp + fp + fn)
    else:
        f1 = 0

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
    # TODO: Implement computing accuracy

    tp = 0

    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            tp += 1
    return tp / len(prediction)
