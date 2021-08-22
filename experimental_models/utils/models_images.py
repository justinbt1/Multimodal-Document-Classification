from tensorflow import keras
import numpy as np


def image_cnn_model(
        filter_map_1=20,
        kernel_size_1=7,
        filter_map_2=50,
        kernel_size_2=5,
        pooling_kernel=4,
        dropout_rate=0.5,
        dense_nodes=1000,
        optimizer='adam',
        loss='categorical_crossentropy'
):
    """ Constructs Image Classification Convolutional Neural Network.

    Args:
        filter_map_1(int): Number of layers in the first 2D convolutional layer.
        kernel_size_1(int): Kernel size for first 2D convolutional layer.
        filter_map_2(int): Number of layers in the second 2D convolutional layer.
        kernel_size_2(int): Kernel size for second 2D convolutional layer.
        pooling_kernel(int): Kernel size for all 2D max pooling layers.
        dropout_rate(float): Dropout rate for final dense hidden layer.
        dense_nodes(int): Number of nodes in fully connected dense layers.
        optimizer(str): Optimization algorithm.
        loss(str): Loss function.

    Returns:
        Image CNN model with defined architecture.

    """
    image_input = keras.layers.Input(shape=(200, 200, 1))
    conv_2d_1 = keras.layers.Conv2D(filter_map_1, kernel_size_1, activation='relu', padding='same')(image_input)
    pool_2d_1 = keras.layers.MaxPooling2D(pooling_kernel)(conv_2d_1)
    conv_2d_2 = keras.layers.Conv2D(filter_map_2, kernel_size_2, activation='relu', padding='valid')(pool_2d_1)
    pool_2d_2 = keras.layers.MaxPooling2D(pooling_kernel)(conv_2d_2)
    extracted_feature = keras.layers.Flatten()(pool_2d_2)
    dense_1 = keras.layers.Dense(dense_nodes, activation='relu')(extracted_feature)
    dense_2 = keras.layers.Dense(dense_nodes, activation='relu')(dense_1)
    dropout_layer = keras.layers.Dropout(dropout_rate)(dense_2)
    output = keras.layers.Dense(6, activation='softmax')(dropout_layer)

    model = keras.models.Model(inputs=[image_input], outputs=[output])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def majority_vote(probabilities):
    """ Majority vote classifier.

    Args:
        probabilities(np.array): Predicted probabilities.

    Returns:
        np.array: Predicted votes.

    """
    start = 0
    end = 10

    predicted_votes = []

    for i in range(int(probabilities.shape[0] / 10)):
        doc_probs = probabilities[start:end]
        classes = []
        for doc in doc_probs:
            classes.append(np.argmax(doc, axis=-1))

        predicted_votes.append(max(set(classes), key=classes.count))

        start += 10
        end += 10

    predicted_votes = np.array(predicted_votes)

    return predicted_votes


def aggregate_probabilities(probabilities):
    """

    Args:
        probabilities:

    Returns:

    """
    start = 0
    end = 10

    aggregate_probs = []

    for i in range(int(probabilities.shape[0] / 10)):
        aggregate_probs.append(probabilities[start:end].flatten())

        start += 10
        end += 10

    aggregate_probs = np.array(aggregate_probs)

    return aggregate_probs


def ann_classifier_model():
    """ Simple Feedforward Network With Single Hidden Layer.

    Returns:
        ANN Neural Network.

    """
    input_layer = keras.layers.Input(shape=60)
    dense1 = keras.layers.Dense(10, activation='relu')(input_layer)
    output = keras.layers.Dense(6, activation='softmax')(dense1)

    ann = keras.models.Model(inputs=[input_layer], outputs=[output])
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return ann


def image_cnn_lstm_model(
        filter_map_1=20,
        kernel_size_1=7,
        filter_map_2=50,
        kernel_size_2=5,
        pooling_kernel=4,
        dropout_rate=0.5,
        lstm_nodes=1000,
        optimizer='adam',
        loss='categorical_crossentropy',
        bi_directional=False,
):
    """ Constructs Image Classification C-LSTM Neural Network.

    Args:
        filter_map_1(int): Number of layers in the first 2D convolutional layer.
        kernel_size_1(int): Kernel size for first 2D convolutional layer.
        filter_map_2(int): Number of layers in the second 2D convolutional layer.
        kernel_size_2(int): Kernel size for second 2D convolutional layer.
        pooling_kernel(int): Kernel size for all 2D max pooling layers.
        dropout_rate(float): Dropout rate for final dense hidden layer.
        lstm_nodes(int): Number of nodes to include in LSTM layers.
        optimizer(str): Optimization algorithm.
        loss(str): Loss function.
        bi_directional(bool): Bi instead of uni directional LSTM layers.

    Returns:
        Image C-LSTM model with defined architecture.

    """
    image_input = keras.layers.Input(shape=(10, 200, 200, 1))

    conv_2d_1 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(filter_map_1, kernel_size_1, activation='relu', padding='same')
    )(image_input)

    pool_2d_1 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pooling_kernel))(conv_2d_1)

    conv_2d_2 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(filter_map_2, kernel_size_2, activation='relu', padding='valid')
    )(pool_2d_1)

    pool_2d_2 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pooling_kernel))(conv_2d_2)

    extracted_features = keras.layers.TimeDistributed(
        keras.layers.Flatten()
    )(pool_2d_2)

    if bi_directional:
        lstm_1 = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_nodes, return_sequences=True)
        )(extracted_features)

        lstm_2 = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_nodes, dropout=dropout_rate)
        )(lstm_1)
    else:
        lstm_1 = keras.layers.LSTM(lstm_nodes, return_sequences=True)(extracted_features)
        lstm_2 = keras.layers.LSTM(lstm_nodes, dropout=dropout_rate)(lstm_1)

    output = keras.layers.Dense(6, activation='softmax')(lstm_2)

    model = keras.models.Model(inputs=[image_input], outputs=[output])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
