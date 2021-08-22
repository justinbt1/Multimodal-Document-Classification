from tensorflow import keras


def early_fusion_model(vocab_length):
    """ Multimodal early fusion model.

    Args:
        vocab_length(int): vocabulary length.

    Returns:
        Multimodal early fusion model

    """
    text_input = keras.layers.Input(shape=(10, 2000), name='text_input')
    embeddings = keras.layers.TimeDistributed(
        keras.layers.Embedding(vocab_length, 150, input_length=2000),
        name='word_embeddings'
    )(text_input)
    conv_1d = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=200, kernel_size=7, activation='relu'),
        name='1d_convolutional_layer'
    )(embeddings)
    global_pooling = keras.layers.TimeDistributed(keras.layers.GlobalMaxPool1D(), name='max_pooling_layer')(conv_1d)
    image_features = keras.layers.TimeDistributed(keras.layers.Flatten(), name='text_features')(global_pooling)

    image_input = keras.layers.Input(shape=(10, 200, 200, 1), name='image_input')
    conv_2d_1 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(20, 7, activation='relu', padding='same'),
        name='2d_convolutional_layer_1'
    )(image_input)
    pool_2d_1 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4), name='2d_max_pooling_layer_1')(conv_2d_1)
    conv_2d_2 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(50, 5, activation='relu', padding='valid'),
        name='2d_convolutional_layer_2'
    )(pool_2d_1)
    pool_2d_2 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4), name='2d_max_pooling_layer_2')(conv_2d_2)
    text_features = keras.layers.TimeDistributed(keras.layers.Flatten(), name='image_features')(pool_2d_2)

    joint_features = keras.layers.concatenate([text_features, image_features])

    lstm_1 = keras.layers.LSTM(450, return_sequences=True)(joint_features)
    lstm_2 = keras.layers.LSTM(1000)(lstm_1)
    dropout = keras.layers.Dropout(0.5)(lstm_2)
    output = keras.layers.Dense(6, activation='softmax')(dropout)

    model = keras.models.Model(inputs=[text_input, image_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def late_fusion_model(vocab_length):
    """ Multimodal late fusion model.

    Args:
        vocab_length(int): vocabulary length.

    Returns:
        Multimodal late fusion model.

    """
    # Text CNN
    text_input = keras.layers.Input(shape=2000)
    embeddings = keras.layers.Embedding(vocab_length, 150, input_length=2000)(text_input)
    conv_1d = keras.layers.Conv1D(filters=200, kernel_size=7, activation='relu')(embeddings)
    global_pooling = keras.layers.GlobalMaxPool1D()(conv_1d)
    flatten = keras.layers.Flatten()(global_pooling)
    dense_layer = keras.layers.Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(0.5))(flatten)
    text_features = keras.layers.Dropout(0.3)(dense_layer)

    # Image CNN LSTM
    image_input = keras.layers.Input(shape=(10, 200, 200, 1))
    conv_2d_1 = keras.layers.TimeDistributed(keras.layers.Conv2D(20, 7, activation='relu', padding='same'))(image_input)
    pool_2d_1 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4))(conv_2d_1)
    conv_2d_2 = keras.layers.TimeDistributed(keras.layers.Conv2D(50, 5, activation='relu', padding='valid'))(pool_2d_1)
    pool_2d_2 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4))(conv_2d_2)
    extracted_features = keras.layers.TimeDistributed(keras.layers.Flatten())(pool_2d_2)
    lstm_1 = keras.layers.LSTM(1000, return_sequences=True)(extracted_features)
    image_features = keras.layers.LSTM(1000, dropout=0.5)(lstm_1)

    # Feed Forward Softmax Classifier
    concat_features = keras.layers.concatenate([text_features, image_features])
    output = keras.layers.Dense(6, activation='softmax')(concat_features)

    model = keras.models.Model(inputs=[text_input, image_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def hybrid_fusion_model(vocab_length):
    """ Multimodal hybrid fusion model.

    Args:
        vocab_length(int): vocabulary length.

    Returns:
        Multimodal hybrid fusion model.

    """
    late_fusion_text_input = keras.layers.Input(shape=2000)
    early_fusion_text_input = keras.layers.Input(shape=(10, 2000))
    image_input = keras.layers.Input(shape=(10, 200, 200, 1))

    lf_embeddings = keras.layers.Embedding(vocab_length, 150, input_length=2000)(late_fusion_text_input)
    lf_conv_1d = keras.layers.Conv1D(filters=200, kernel_size=7, activation='relu')(lf_embeddings)
    lf_global_pooling = keras.layers.GlobalMaxPool1D()(lf_conv_1d)
    lf_flatten = keras.layers.Flatten()(lf_global_pooling)
    lf_dense_layer = keras.layers.Dense(
        50, activation='relu', kernel_regularizer=keras.regularizers.l2(0.5)
    )(lf_flatten)
    lf_text_features = keras.layers.Dropout(0.3)(lf_dense_layer)

    lf_conv_2d_1 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(20, 7, activation='relu', padding='same')
    )(image_input)
    lf_pool_2d_1 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4))(lf_conv_2d_1)
    lf_conv_2d_2 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(50, 5, activation='relu', padding='valid')
    )(lf_pool_2d_1)
    lf_pool_2d_2 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4))(lf_conv_2d_2)
    image_features = keras.layers.TimeDistributed(keras.layers.Flatten())(lf_pool_2d_2)
    lf_lstm_1 = keras.layers.LSTM(1000, return_sequences=True)(image_features)
    lf_image_features = keras.layers.LSTM(1000, dropout=0.5)(lf_lstm_1)

    lf_merge_features = keras.layers.concatenate([lf_text_features, lf_image_features])
    late_fusion_features = keras.layers.Flatten()(lf_merge_features)

    ef_embeddings = keras.layers.TimeDistributed(
        keras.layers.Embedding(vocab_length, 150, input_length=2000),
        name='word_embeddings'
    )(early_fusion_text_input)
    ef_conv_1d = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=200, kernel_size=7, activation='relu'),
        name='1d_convolutional_layer'
    )(ef_embeddings)
    ef_global_pooling = keras.layers.TimeDistributed(
        keras.layers.GlobalMaxPool1D(),
        name='max_pooling_layer'
    )(ef_conv_1d)
    ef_text_features = keras.layers.TimeDistributed(keras.layers.Flatten(), name='text_features')(ef_global_pooling)

    ef_conv_2d_1 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(20, 7, activation='relu', padding='same')
    )(image_input)
    ef_pool_2d_1 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4))(ef_conv_2d_1)
    ef_conv_2d_2 = keras.layers.TimeDistributed(keras.layers.Conv2D(50, 5, activation='relu', padding='valid'))(
        ef_pool_2d_1)
    ef_pool_2d_2 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(4))(ef_conv_2d_2)
    ef_image_features = keras.layers.TimeDistributed(keras.layers.Flatten())(ef_pool_2d_2)

    ef_joint_features = keras.layers.concatenate([ef_text_features, ef_image_features])

    ef_lstm_1 = keras.layers.LSTM(450, return_sequences=True)(ef_joint_features)
    ef_lstm_2 = keras.layers.LSTM(1000)(ef_lstm_1)
    early_fusion_features = keras.layers.Dropout(0.5)(ef_lstm_2)

    hybrid_representation = keras.layers.concatenate([late_fusion_features, early_fusion_features])
    output = keras.layers.Dense(6, activation='softmax')(hybrid_representation)

    model = keras.models.Model(inputs=[late_fusion_text_input, early_fusion_text_input, image_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
