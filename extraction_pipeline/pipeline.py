import os
import json
import multiprocessing

# Project modules.
import text_utils
import database_utils
import file_parsers


class Config:
    """ Creates object containing config file parameters.

    Attributes:
        file_directory(str): Directory containing original files.
        image_output_dir(str): Output directory for converted images.
        text_output_dir(str): Output directory for extracted text.
        self.n_cores(int): Number of cores to use for multiprocessing.

    """
    def __init__(self):
        config_file = open('../configs/config.json', 'rt')
        config = json.load(config_file)
        config_file.close()

        self.file_directory = config['File Directory']
        self._validate_directory(self.file_directory)

        self.image_output_dir = config['Image Output Directory']
        self._validate_directory(self.image_output_dir)

        self.n_page_images = config['N Page Images']

        self.text_output_dir = config['Text Output Directory']
        self._validate_directory(self.text_output_dir)

        self.tika_server_path = config['Tika Server']

        self.n_cores = config['N Cores']

    @staticmethod
    def _validate_directory(directory):
        """ Checks directory path is valid.

        Args:
            directory(str): Directory path.

        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(
                f'Text output path {directory} is not a valid directory.'
            )


class FileProperties:
    """ Tracks extraction details for a file.

    Attributes:
        file_path(str): Path of original file.
        text_path(str): Path for output text JSON file.
        image_path(str): Path for output directory for JPEG files.
        image_extracted(int): Was image extraction successful.
        text_extracted(int): Was text extraction successful.
        tika_status(int): Tika status code.

    """
    def __init__(self, file_path, text_path, image_path, n_page_images):
        """ Tracks extraction details for a file.

        Args:
            file_path(str): Path of original file.
            text_path(str): Path for output text JSON file.
            image_path(str): Path for output directory for JPEG files.

        """
        self.file_path = file_path

        split_path = file_path.split('\\')
        self.filename = split_path[-1]

        filename_base, ext = os.path.splitext(self.filename)
        self.ext = ext.lower()

        self.label = split_path[-2].lower()
        self.text_path = os.path.join(text_path, f'{filename_base}.json')
        self.image_path = os.path.join(image_path, filename_base)

        self.n_page_images = n_page_images
        self.error = None

        self.image_extracted = 0
        self.ocr = 0
        self.text_extracted = 0
        self.tika_status = 0


def extraction_process(queue):
    """ Handles a processes queue monitoring, database and calls extraction process.

    Args:
        queue(multiprocessing.queue): Multiprocessing queue object.

    """
    database = database_utils.Database()
    database.connect()

    while True:
        if queue.empty():
            break

        file_properties = queue.get()
        print(file_properties.file_path)

        image_exts = ('.tif', '.tiff', '.jpeg', '.png', '.gif')

        text_exts = (
            '.txt', '.las', '.asc', '.ascii', '.dat', '.lst', '.geo', '.csv', '.prn', '.xyz', '.p190', '.tfw'
        )

        try:
            if file_properties.ext == '.pdf':
                file_properties = file_parsers.parse_pdf_file(file_properties)
            elif file_properties.ext in image_exts:
                file_properties = file_parsers.parse_image_file(file_properties)
            elif file_properties.ext in text_exts:
                file_properties = file_parsers.parse_text_file(file_properties)

        except Exception as e:
            file_properties.error = str(e)

        database.insert(file_properties)

    database.connection.close()


def pipeline():
    """ Lists files and manages pipeline processes and flow.

    """
    config = Config()
    tika_server = text_utils.start_tika_server(config.tika_server_path)

    if config.n_cores == -1:
        config.n_cores = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(config.n_cores)
    process_manager = multiprocessing.Manager()
    queue = process_manager.Queue()

    for root, dirs, files in os.walk(config.file_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_properties = FileProperties(
                file_path,
                config.text_output_dir,
                config.image_output_dir,
                config.n_page_images
            )

            queue.put(file_properties)

    for core in range(config.n_cores):
        pool.apply_async(extraction_process, args=[queue])

    pool.close()
    pool.join()

    tika_server.kill()


if __name__ == '__main__':
    pipeline()
