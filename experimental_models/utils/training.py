import pickle
import sklearn
import numpy as np
from tensorflow import keras


def iterate_training(model, data, iterations, y, data_type, keep_weights=False, keep_preds=False, model_params=None):
    """

    Args:
        model: Keras model to be evaluated.
        data: Data object.
        iterations(int): Number of iterations to perform.
        y(np.array): Y class labels.
        data_type(bool): Text or image data.
        keep_weights(bool): Keep model weights.
        keep_preds(bool): Keep model predictions.
        model_params(dict): Model hyper parameter kwargs.

    Returns:
        tuple: Average performance metrics.

    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    losses = []
    weights = []
    predictions = []

    # Avoid immutable default in params.
    if not model_params:
        model_params = {}

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min',
        restore_best_weights=True
    )

    if data_type == 'image':
        x_train = data.image_train
        x_test = data.image_test
        x_val = data.image_val
    elif data_type == 'text':
        x_train = data.text_train
        x_test = data.text_test
        x_val = data.text_val
    else:
        raise ValueError('data_type must be "image" or "text"')

    for iteration in range(iterations):
        print(f'---- Training model iteration {iteration}:')

        eval_model = model(**model_params)

        eval_model.fit(
            x_train,
            data.y_train,
            epochs=100,
            validation_data=(x_val, data.y_val),
            callbacks=[early_stopping],
            verbose=2
        )

        loss, accuracy = eval_model.evaluate(x_test, data.y_test)

        losses.append(loss)
        accuracy_scores.append(accuracy)

        y_probs = eval_model.predict(x_test)
        predictions.append(y_probs)
        y_hat = np.argmax(y_probs, axis=-1)

        precision = sklearn.metrics.precision_score(y, y_hat, average='macro')
        recall = sklearn.metrics.recall_score(y, y_hat, average='macro')
        f1_score = sklearn.metrics.f1_score(y, y_hat, average='macro')

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

        print(
            f'Metrics at convergence: Loss: {loss}, accuracy: {accuracy}, f1 score: {f1_score}\n'
        )

        if keep_weights:
            weights.append(eval_model.get_weights())

        # Clear GPU memory before proceeding with next evaluation run.
        keras.backend.clear_session()

    metrics = {
        'ave_precision': sum(precision_scores) / len(precision_scores),
        'ave_recall': sum(recall_scores) / len(recall_scores),
        'ave_f1_score': sum(f1_scores) / len(f1_scores),
        'ave_accuracy': sum(accuracy_scores) / len(accuracy_scores),
        'ave_loss': sum(losses) / len(losses)
    }

    output = [metrics]

    if keep_weights:
        output.append(weights)

    if keep_preds:
        output.append(predictions)

    return tuple(output)


def get_weights():
    """ Loads uni-modal text and image CNN model weights.

    Returns:
        tuple: text and image weights.

    """
    text_weight_file = open("models/unimodal_text_CNN_weights.pickle", "rb")
    text_weights = pickle.load(text_weight_file)
    text_weight_file.close()

    image_weight_file = open("models/unimodal_image_CNN_LSTM_weights.pickle", "rb")
    image_weights = pickle.load(image_weight_file)
    image_weight_file.close()

    return text_weights, image_weights
