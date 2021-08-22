import os
import json
import PIL
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

from utils.database import get_db_table


class DocumentData:
    """ Loads and prepares text and image data prior to modelling.

    Attributes:
        seed(int): Seed for random processes.
        n_classes(int): Number of classes present in dataset.
        vocab_length(int): Number of unique terms in vocabulary.
        text_train(np.array): Text training tokens.
        text_val(np.array): Text validation tokens.
        text_test(np.array): Text test tokens.
        image_train(np.array): Sequence of n training images.
        image_val(np.array): Sequence of n validation images.
        image_test(np.array): Sequence of n test images.
        y_train(np.array): Training class labels.
        y_val(np.array): Validation class labels.
        y_test(np.array): Test class labels.

    """
    def __init__(self, label_map, seed, test_size=0.2, validation_size=0.2, drop_nans=False):
        """ TextData object constructor.

        Args:
             label_map(dict): label_map(dict): Mapping of labels to numbers.
             seed(int): Seed for random processes.
             test_size(float): Size of test hold out set.
             validation_size(float): Size of validation set.
             drop_nans(bool): Drop nan data?

        """
        self.seed = seed
        self.n_classes = len(label_map)
        data_frame = get_db_table()
        data_frame.drop_duplicates(['image_dir_ref', 'text_json_ref'], inplace=True)

        if drop_nans:
            if drop_nans == 'all':
                data_frame = data_frame.loc[
                    (data_frame['image_extracted'] == 1) & (data_frame['text_extracted'] == 1)
                    ]
            elif drop_nans == 'text':
                data_frame = data_frame.loc[data_frame['text_extracted'] == 1]
            elif drop_nans == 'image':
                data_frame = data_frame.loc[data_frame['image_extracted'] == 1]
            elif drop_nans == 'or':
                data_frame = data_frame.loc[
                    (data_frame['image_extracted'] == 1) | (data_frame['text_extracted'] == 1)
                    ]
            else:
                raise ValueError(
                    'drop_nans parameter is invalid please use False, "all", "text", "image" or "Or"'
                )

        labels = np.array([label_map[label] for label in data_frame['label']])

        x_train, x_test, y_train, y_test = train_test_split(
            data_frame.index,
            labels,
            test_size=test_size,
            train_size=1.0 - test_size,
            random_state=seed,
            shuffle=True,
            stratify=labels
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=validation_size,
            train_size=1.0 - validation_size,
            random_state=seed,
            shuffle=True,
            stratify=y_train
        )

        self._x_train = data_frame.loc[x_train]
        self._x_val = data_frame.loc[x_val]
        self._x_test = data_frame.loc[x_test]

        self.vocab_length = 0

        self.text_train = None
        self.text_val = None
        self.text_test = None

        self.image_train = None
        self.image_val = None
        self.image_test = None

        self.y_train = keras.utils.to_categorical(y_train, num_classes=self.n_classes)
        self.y_val = keras.utils.to_categorical(y_val, num_classes=self.n_classes)
        self.y_test = keras.utils.to_categorical(y_test, num_classes=self.n_classes)

        self.data_frame = data_frame

    def load_text_data(self, text_length=2000):
        """ Loads text and processes sequences for modelling.

        Args:
            text_length(int): Maximum number of words in sequence.

        """
        train_text = self._load_processed_text(self._x_train['text_json_ref'])
        val_text = self._load_processed_text(self._x_val['text_json_ref'])
        test_text = self._load_processed_text(self._x_test['text_json_ref'])

        tokenizer = keras.preprocessing.text.Tokenizer(oov_token=1, split=' ')
        tokenizer.fit_on_texts(train_text)
        self.vocab_length = len(tokenizer.word_index) + 1

        train_text = tokenizer.texts_to_sequences(train_text)
        val_text = tokenizer.texts_to_sequences(val_text)
        test_text = tokenizer.texts_to_sequences(test_text)

        self.text_train = keras.preprocessing.sequence.pad_sequences(
            train_text,
            maxlen=text_length,
            padding='post',
            truncating='post',
            value=0
        )

        self.text_val = keras.preprocessing.sequence.pad_sequences(
            val_text,
            maxlen=text_length,
            padding='post',
            truncating='post',
            value=0
        )

        self.text_test = keras.preprocessing.sequence.pad_sequences(
            test_text,
            maxlen=text_length,
            padding='post',
            truncating='post',
            value=0
        )

    @staticmethod
    def _load_processed_text(db_series):
        """ Loads processed text from JSON output files.

        Args:
            db_series(pd.Series): Data series containing JSON file paths.

        Returns:
            list: List of extracted text strings.

        """
        texts = []

        for text_path in db_series:
            text = ''

            if os.path.exists(text_path):
                text_file = open(text_path, 'rt')
                text_json = json.load(text_file)
                text_file.close()

                text = text_json['Clean Content']

                if text:
                    text = ' '.join(text.split()[0:2000])

            texts.append(text)

        return texts

    def load_image_data(self, image_size=200, n_pages=10, sequential=False):
        """ Returns page image sequences for all documents.

        Args:
            image_size(int): Image size, assumes image is a square.
            n_pages(int): Number of page images in sequence.
            sequential(bool): Return all pages for all documents as a single sequence.

        Returns:
            tuple: page image sequences for all documents, labels.

        """
        if image_size > 500:
            raise ValueError('Image size cannot exceed 500!')

        self.image_train = self._load_document_images(
            self._x_train['image_dir_ref'], image_size, n_pages, sequential
        )

        self.image_val = self._load_document_images(
            self._x_val['image_dir_ref'], image_size, n_pages, sequential
        )

        self.image_test = self._load_document_images(
            self._x_test['image_dir_ref'], image_size, n_pages, sequential
        )

    @staticmethod
    def _data_paths(sample_directories, n_pages):
        """ Get all image sequence paths.

        Args:
            sample_directories(np.array): Document image paths.
            n_pages(int): Number of page images in sequence.

        Returns:
            dict: Document, image path pairs.

        """
        document_paths = {}
        for directory in sample_directories:
            if not os.path.exists(directory):
                document_paths[directory] = []
                continue

            page_images = os.listdir(directory)[0:n_pages]

            if not page_images:
                document_paths[directory] = []
                continue

            document_paths[directory] = page_images
        return document_paths

    def _load_document_images(self, directory_paths, image_size, n_pages, sequential):
        """ Loads document images from document image directories.

        Args:
            directory_paths(np.array): List of directories containing page images.
            image_size(int): Image size, assumes image is a square.
            n_pages(int): Number of page images in sequence.
            sequential(bool): Return all pages for all documents as a single sequence.

        Returns:
            np.array: 2D or 3D array containing the page images of each document.

        """
        document_paths = self._data_paths(directory_paths, n_pages)

        x = []

        for document_path in document_paths:
            image_paths = document_paths[document_path]

            image_sequence = self._process_sequences(
                document_path, image_paths, image_size, n_pages
            )

            if n_pages == 1:
                image_sequence = image_sequence[0]

            x.append(image_sequence)

        x = np.array(x)
        x = x / 255

        if n_pages == 1:
            x = x.reshape((x.shape[0], image_size, image_size, 1))
        elif sequential:
            x = x.reshape((x.shape[0] * n_pages, image_size, image_size, 1))
        else:
            x = x.reshape((x.shape[0], n_pages, image_size, image_size, 1))

        return x

    @staticmethod
    def _process_sequences(dir_path, image_paths, image_size, n_pages):
        """ Loads sequence of page images as array.

        Args:
            dir_path(str): Document image directory path.
            image_paths(list): Paths to page images.
            image_size(int): Image size, assumes image is a square.
            n_pages(int): Number of page images in sequence.

        Returns:
            np.array: Array of image sequences.

        """
        image_sequence = []
        for page_image in image_paths:
            image_path = os.path.join(dir_path, page_image)
            image = PIL.Image.open(image_path).convert('L')
            image = image.resize((image_size, image_size))
            image_sequence.append(np.array(image))

        sequence_length = len(image_sequence)

        if sequence_length < n_pages:
            n_blank_images = n_pages - sequence_length
            image_sequence += [np.zeros((image_size, image_size))] * n_blank_images

        image_sequence = np.array(image_sequence)

        return image_sequence
