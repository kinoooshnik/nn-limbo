def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    correct = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            correct += 1

    if prediction.shape[0] != 0:
        accuracy = correct / prediction.shape[0]
    else:
        accuracy = 0
    return accuracy