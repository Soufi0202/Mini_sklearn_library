import numpy as np
import matplotlib.pyplot as plt

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    """Calculate the precision of the predictions, defined as tp / (tp + fp)."""
    true_positives = np.sum((y_pred == 1) & (y_true == y_pred))
    predicted_positives = np.sum(y_pred == 1)
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    return precision

def recall_score(y_true, y_pred):
    """Calculate the recall of the predictions, defined as tp / (tp + fn)."""
    true_positives = np.sum((y_pred == 1) & (y_true == y_pred))
    actual_positives = np.sum(y_true == 1)
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    return recall

def f1_score(y_true, y_pred):
    """Calculate the F1 score, the harmonic mean of precision and recall."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return f1




def confusion_matrix(y_true, y_pred, labels=None):

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_index[true_label]
        pred_idx = label_to_index[pred_label]
        cm[true_idx, pred_idx] += 1

    return cm




def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
    cm : confusion matrix to be plotted.
    labels : array-like of shape (n_classes), list of labels to index the matrix.
    title : string, title of the confusion matrix.
    cmap : colormap, optional (default=plt.cm.Blues), colormap to be used.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


