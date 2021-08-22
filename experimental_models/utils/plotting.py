import seaborn as sns
from matplotlib import pyplot as plt
import sklearn


def plot_history(history, title):
    """ Plots model history.

    Args:
        history: Model training history.
        title(str): Plot title.

    """
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} - Classification Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} - Crossentropy Loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def confusion_matrix(y, y_hat, label_map, title):
    """ Plots confusion matrix.

    Args:
        y(np.array): Ground truth classes.
        y_hat(np.array): Predicted classes.
        label_map(dict): Mapping between text and numeric label.
        title(str): Plot title.

    """
    matrix = sklearn.metrics.confusion_matrix(y, y_hat)

    plt.figure(figsize=(14, 8))
    labels = [label for label in label_map]
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        fmt='d',
        cmap='Blues'
    )
    plt.yticks(rotation=0)
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
