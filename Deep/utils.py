import numpy as np
from sklearn.metrics import matthews_corrcoef


def accuracy(labels, predictions):
    predictions = np.argmax(predictions, axis=1).flatten()
    labels = labels.flatten()
    size = len(labels)

    return np.sum(predictions == labels) / size

def mcc(labels, predictions):
    labels = np.concatenate(labels, axis=0)

    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1).flatten()    

    return matthews_corrcoef(labels, predictions)
