import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    TP_or_TN = np.sum((prediction[i] == ground_truth[i]) for i in range(prediction.shape[0]))
    accuracy = TP_or_TN / len(prediction)

    return accuracy
