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

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for i in range(len(prediction)): 
        if ground_truth[i] == prediction[i] == True:
            TP += 1
        if prediction[i] == True and ground_truth[i] == False:
            FP += 1
        if ground_truth[i] == prediction[i] == False:
            TN += 1
        if prediction[i] == False and ground_truth[i] == True:
            FN += 1
            
    if (TP + FP) != 0:        
        precision = (TP) / (TP + FP)
    else:
        precision = 0    
    if (TP + FN) != 0:
        recall = (TP) / (TP + FN)
    else:
        recall = 0
    if prediction.shape[0] != 0: 
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        accuracy = 0  
    if (precision + recall) != 0:    
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
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
    accuracy = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            accuracy += 1
 
    return accuracy / len(prediction)
