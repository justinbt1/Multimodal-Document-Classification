from tensorflow import keras


def text_cnn_model(
        doc_data,
        embedding_size=150,
        filter_maps=100,
        kernel_size=4,
        dropout_rate=0.5,
        l2_regularization=3,
        dense_nodes=100,
        dense_layers=0,
        optimizer='adam',
        loss='categorical_crossentropy'
):
    """ Constructs text Classification Convolutional Neural Network.

    Args:
        doc_data: Object containing text data.
        embedding_size(float): Size of text embeddings.
        filter_maps(int): Number of filter maps for 1D CNN layer.
        kernel_size(int): Kernel size for 1D CNN layer.
        dropout_rate(float): Dropout rate for final dense fully connected classifier layer.
        l2_regularization(float): Rate of l2 regularization.
        dense_nodes(int): Number of nodes in each dense fully connected classifier layer.
        dense_layers(int): Number of dense fully connected classifier layers.
        optimizer(str): Optimization algorithm.
        loss(str): Loss function.

    Returns:
        Text CNN model with defined architecture.

    """
    input_layer = keras.layers.Input(shape=2000)
    embeddings = keras.layers.Embedding(doc_data.vocab_length, embedding_size, input_length=2000)(input_layer)
    conv_1d = keras.layers.Conv1D(filters=filter_maps, kernel_size=kernel_size, activation='relu')(embeddings)
    global_pooling = keras.layers.GlobalMaxPool1D()(conv_1d)
    extracted_features = keras.layers.Flatten()(global_pooling)

    if dense_layers > 0:
        dense_layer = keras.layers.Dense(dense_nodes, activation='relu')(extracted_features)
        for layer in range(dense_layers - 1):
            dense_layer = keras.layers.Dense(dense_nodes, activation='relu')(dense_layer)
        dropout_layer = keras.layers.Dropout(dropout_rate)(dense_layer)
    else:
        dropout_layer = keras.layers.Dropout(dropout_rate)(extracted_features)

    regularise = keras.regularizers.l2(l2_regularization)
    output = keras.layers.Dense(6, activation='softmax', kernel_regularizer=regularise)(dropout_layer)

    model = keras.models.Model(inputs=[input_layer], outputs=[output])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
