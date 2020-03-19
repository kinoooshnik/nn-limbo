def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # DONE: implement metrics!
    
    num_samples = prediction.shape[0]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(num_samples):
        if (ground_truth[i]):
            if (prediction[i]):
                TP += 1
            else:
                FN += 1
        else:
            if (prediction[i]):
                FP += 1
            else:
                TN += 1
    
    precision = 0
    recall = 0
    f1 = 0
    accuracy = 0
    
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    if (TP + FN) != 0:        
        recall = TP / (TP + FN)
    if (TP + FN + TN + FP) != 0:
        accuracy = (TP + TN) / (TP + FN + TN + FP)
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    num_samples = prediction.shape[0]
    
    if num_samples == 0:
        return 0
    
    TP_TN = 0
    for i in range(num_samples):
        if (ground_truth[i] == prediction[i]):
            TP_TN += 1
            
    accuracy = TP_TN / num_samples
    return accuracy
