from collections import defaultdict
import sklearn
import numpy as np
from tensorflow import keras

from utils import models_images


def evaluation_metrics(metrics_dict, key_prefix, y, y_hat):
    """ Calculates metrics for model evaluation.

    Args:
        metrics_dict(dict): Dictionary to update with metrics.
        key_prefix(str): Dictionary key prefix.
        y(np.array): True value.
        y_hat(np.array): Predicted value.

    Returns:
        dict: Updated metrics dictionary.

    """
    metrics_dict[f'{key_prefix}_accuracy'].append(sklearn.metrics.accuracy_score(y, y_hat, average='macro'))
    metrics_dict[f'{key_prefix}_precision'].append(sklearn.metrics.precision_score(y, y_hat, average='macro'))
    metrics_dict[f'{key_prefix}_recall'].append(sklearn.metrics.recall_score(y, y_hat, average='macro'))
    metrics_dict[f'{key_prefix}_score'].append(sklearn.metrics.f1_score(y, y_hat, average='macro'))

    return metrics_dict


def multi_y(labels):
    """ Convert y values to a sequence.

    Args:
        labels(np.array): y values.

    Returns:
        np.array: y value sequence.

    """
    ys = []

    for y in labels:
        for i in range(10):
            ys.append(y)

    ys = np.array(ys)
    return ys


def evaluate_ann(evaluation, y_class, prob_train, prob_val, prob_test, data, early_stopping):
    """ Evaluate ANN approach to multi-image ensemble classification.

    Args:
        evaluation(dict): Dictionary of evaluation metrics.
        y_class(np.array): Discrete test classes.
        prob_train(np.array): Training probabilities.
        prob_val(np.array): Training probabilities.
        prob_test(np.array): Training probabilities.
        data: Data object.
        early_stopping(keras.callbacks.EarlyStopping): Early stopping criteria.

    Returns:
        dict: Updated evaluation metrics dictionary.

    """
    # Concat page probabilities prior to ANN
    ann_train = models_images.aggregate_probabilities(prob_train)
    ann_val = models_images.aggregate_probabilities(prob_val)
    ann_test = models_images.aggregate_probabilities(prob_test)

    ann = models_images.ann_classifier_model()
    ann.fit(ann_train, data.y_train, epochs=400, validation_data=(ann_val, data.y_val), callbacks=[early_stopping])

    ann_y_probs = ann.predict(ann_test)
    ann_y_hat = np.argmax(ann_y_probs, axis=-1)

    evaluation = evaluation_metrics(evaluation, 'ann', y_class, ann_y_hat)

    return evaluation


def evaluate_single_iteration(evaluation, data, seq_y_train, seq_y_val, y_class, early_stopping):
    """

    Args:
        evaluation(dict): Dictionary of evaluation metrics.
        data: Data object.
        seq_y_train(np.array): Y value training sequence.
        seq_y_val(np.array): Y value validation sequence.
        y_class(np.array): Discrete test classes.
        early_stopping(keras.callbacks.EarlyStopping): Early stopping criteria.

    Returns:
        dict: Evaluation metrics for each iteration.

    """
    cnn = models_images.image_cnn_model()
    cnn.fit(
        data.image_train,
        seq_y_train,
        epochs=400,
        validation_data=(data.image_val, seq_y_val),
        callbacks=[early_stopping]
    )

    # Inference on training, validation, test data.
    prob_train = cnn.predict(data.image_train)
    prob_val = cnn.predict(data.image_val)
    prob_test = cnn.predict(data.image_test)

    # Get predicted page classes and evaluation metrics.
    predicted_probs = np.argmax(prob_test, axis=-1)
    evaluation = evaluation_metrics(evaluation, 'page', y_class, predicted_probs)

    # Calculate majority vote and evaluation metrics.
    test_vote = models_images.majority_vote(prob_test)
    evaluation = evaluation_metrics(evaluation, 'vote', y_class, test_vote)

    # Calculate ANN evaluation metrics.
    evaluation = evaluate_ann(evaluation, y_class, prob_train, prob_val, prob_test, data, early_stopping)

    return evaluation


def optimise_multi_page_cnn(data, iterations):
    """ Iteratively evaluate multi page image CNN model.

    Args:
        data: Data object.
        iterations(int): Number of iterations.

    Returns:
        dict: Evaluation metrics.

    """
    evaluation = defaultdict(list)

    # Convert y values to sequence.
    seq_y_train = multi_y(data.y_train)
    seq_y_val = multi_y(data.y_val)

    # Get discrete test classes.
    y_class = np.argmax(data.y_test, axis=-1)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min',
        restore_best_weights=True
    )

    for iteration in range(iterations):
        evaluation = evaluate_single_iteration(evaluation, data, seq_y_train, seq_y_val, y_class, early_stopping)

    for metric, values in evaluation.items():
        evaluation[metric] = sum(evaluation[metric]) / len(evaluation[metric])

    return evaluation
