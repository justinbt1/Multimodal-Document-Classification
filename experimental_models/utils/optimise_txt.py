from tensorflow import keras

from utils import models_text
from utils import logger


def evaluate_text_model(data, session_num, model_params, early_stopping, log):
    """ Evaluate performance of text model for given hyper parameters.

    Args:
        data: Data object.
        session_num(int): Session tracking number.
        model_params(dict): Model parameters kwargs dict.
        early_stopping(keras.callbacks.EarlyStopping): Early stopping criteria.
        log(logger.Logging): Logging object.

    """
    model = models_text.text_cnn_model(data, **model_params)

    model.fit(
        data.text_train,
        data.y_train,
        epochs=100,
        validation_data=(data.text_val, data.y_val),
        callbacks=[early_stopping]
    )

    loss, accuracy = model.evaluate(data.text_test, data.y_test)

    keras.backend.clear_session()

    log_metrics = [
        str(session_num),
        model_params['filter_maps'],
        model_params['kernel_size'],
        model_params['dropout_rate'],
        model_params['l2_regularization'],
        str(accuracy),
        str(loss)
    ]

    log.write(', '.join(log_metrics) + '\n')

    return accuracy


def text_grid_search(data, filter_regions, feature_maps, dropout_rates, l2_norm_constraints, log_location):
    """ Performs grid search for text model.

    Args:
        data: Data object.
        filter_regions(list): List of filter region counts to evaluate.
        feature_maps(list): List of feature maps to evaluate.
        dropout_rates(list): List of dropout rates to evaluate.
        l2_norm_constraints(list): List of l2 norm constraint values to evaluate.
        log_location(str): Log file path.

    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min',
        restore_best_weights=True
    )

    session_num = 0
    optimal_filter_region = 0
    optimal_feature_map = 0
    optimal_dropout = 0
    optimal_l2_norms = 0
    max_accuracy = 0

    log = logger.Logging(log_location, 'Run, Kernel Size, Feature Maps, Dropout, L2 Norm, Accuracy, Loss\n')
    for filter_region in filter_regions:
        for feature_map in feature_maps:
            session_num += 1

            model_parameters = {
                'filter_maps': feature_map,
                'kernel_size': filter_region,
                'dropout_rate': '0.5',
                'l2_regularization': '3.0'
            }

            accuracy = evaluate_text_model(data, session_num, model_parameters, early_stopping, log)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                optimal_filter_region = filter_region
                optimal_feature_map = feature_map

    for dropout_rate in dropout_rates:
        session_num += 1

        model_parameters = {
            'filter_maps': optimal_feature_map,
            'kernel_size': optimal_filter_region,
            'dropout_rate': dropout_rate,
            'l2_regularization': '3.0'
        }

        accuracy = evaluate_text_model(data, session_num, model_parameters, early_stopping, log)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_dropout = dropout_rate

    for l2_norm_constraint in l2_norm_constraints:
        session_num += 1

        model_parameters = {
            'filter_maps': optimal_feature_map,
            'kernel_size': optimal_filter_region,
            'dropout_rate': optimal_dropout,
            'l2_regularization': l2_norm_constraint
        }

        accuracy = evaluate_text_model(data, session_num, model_parameters, early_stopping, log)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_l2_norms = l2_norm_constraint

    optimal_parameters = {
        'filter_maps': optimal_feature_map,
        'kernel_size': optimal_filter_region,
        'dropout_rate': optimal_dropout,
        'l2_regularization': optimal_l2_norms
    }

    return optimal_parameters
