import os
import json
import numpy as np
from sklearn import metrics
from tensorflow import keras

# Project Modules
from utils import data, models_multimodal


def performance_metrics(y, y_hat):
    perf_metrics = {
        'accuracy': metrics.accuracy_score(y, y_hat),
        'precision': metrics.precision_score(y, y_hat, average='macro'),
        'recall': metrics.recall_score(y, y_hat, average='macro'),
        'f1_score': metrics.f1_score(y, y_hat, average='macro')
    }

    return perf_metrics


def log_performance(perf_metrics, name):
    log_path = os.path.join('performance_logs', f'{name}.json')
    log_file = open(log_path, 'wt')
    json.dump(perf_metrics, log_file, indent=4)
    log_file.close()


def benchmark_performance(model, name, x_train, y_train, x_val, y_val, x_test, y_test, callbacks):
    print(name)
    model.fit(x_train, y_train, epochs=400, validation_data=(x_val, y_val), callbacks=callbacks)
    y_hat = model.predict(x_test)
    y_hat = np.argmax(y_hat, axis=-1)
    perf_metrics = performance_metrics(y_test, y_hat)
    log_performance(perf_metrics, name)
    print(perf_metrics)

    keras.backend.clear_session()


def evaluate_models(data_type, iterations, model, data_seed):
    label_map = {
        'geol_geow': 0,
        'geol_sed': 1,
        'gphys_gen': 2,
        'log_sum': 3,
        'pre_site': 4,
        'vsp_file': 5
    }

    doc_data = data.DocumentData(label_map, data_seed, drop_nans=data_type)
    doc_data.load_text_data()
    doc_data.load_image_data(image_size=200, n_pages=10)

    text_train_seq = np.array([np.array([text] * 10) for text in doc_data.text_train])
    text_val_seq = np.array([np.array([text] * 10) for text in doc_data.text_val])
    text_test_seq = np.array([np.array([text] * 10) for text in doc_data.text_test])

    y = np.argmax(doc_data.y_test, axis=-1)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min',
        restore_best_weights=True
    )

    for iteration in range(iterations):

        # Evaluate Early Fusion Model
        if model == 'early':
            name = f'Early_fusion_eval_{iteration}_{data_type}_{data_seed}'
            print(name)
            early_fusion = models_multimodal.early_fusion_model(doc_data.vocab_length)
            benchmark_performance(
                early_fusion,
                name,
                [text_train_seq, doc_data.image_train],
                doc_data.y_train,
                [text_val_seq, doc_data.image_val],
                doc_data.y_val,
                [text_test_seq, doc_data.image_test],
                y,
                [early_stopping]
            )

        # Evaluate Late Fusion Model
        elif model == 'late':
            name = f'Late_fusion_eval_{iteration}_{data_type}_{data_seed}'
            print(name)
            late_fusion = models_multimodal.late_fusion_model(doc_data.vocab_length)
            benchmark_performance(
                late_fusion,
                name,
                [doc_data.text_train, doc_data.image_train],
                doc_data.y_train,
                [doc_data.text_val, doc_data.image_val],
                doc_data.y_val,
                [doc_data.text_test, doc_data.image_test],
                y,
                [early_stopping]
            )

        # Evaluate Hybrid Fusion Model
        elif model == 'hybrid':
            name = f'Hybrid_fusion_eval_{iteration}_{data_type}_{data_seed}'
            print(name)
            hybrid_fusion = models_multimodal.hybrid_fusion_model(doc_data.vocab_length)
            benchmark_performance(
                hybrid_fusion,
                name,
                [doc_data.text_train, text_train_seq, doc_data.image_train],
                doc_data.y_train,
                [doc_data.text_val, text_val_seq, doc_data.image_val],
                doc_data.y_val,
                [doc_data.text_test, text_test_seq, doc_data.image_test],
                y,
                [early_stopping]
            )


if __name__ == '__main__':
    seeds = (2026, 2027, 2028, 2029)
    for i, seed in enumerate(seeds):
        evaluate_models('all', 1, 'early', seed)
        evaluate_models('all', 1, 'late', seed)
        evaluate_models('all', 1, 'hybrid', seed)
        evaluate_models('or', 1, 'early', seed)
        evaluate_models('or', 1, 'late', seed)
        evaluate_models('or', 1, 'hybrid', seed)
