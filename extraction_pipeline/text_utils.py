import subprocess
import requests
import json
import nltk


def start_tika_server(tika_path):
    """ Launches Tika Server and performs uptime checks.

    Args:
        tika_path(str): Path to Apache Tika Server executable.

    Returns:
        subprocess.Popen: Process object.

    """
    command = f'java -cp "{tika_path}" org.apache.tika.server.TikaServerCli --port 80 --host 127.0.0.1'
    tika_server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        requests.get('http://127.0.0.1:80/tika')

    except requests.exceptions.ConnectionError as e:
        raise SystemExit(f'Unable to connect to Tika Server. {e}')

    return tika_server


def text_processing(text):
    """ Cleans string for use in machine learning.

    Args:
        text(str): Raw text for processing.

    Returns:
        str: Cleaned text.

    """
    lemmatizer = nltk.stem.WordNetLemmatizer()

    text = text.casefold()
    text = text.translate(
        text.maketrans({'\'': None, '-': ' '})
    )

    text = nltk.word_tokenize(text)
    clean_tokens = []

    for token in text:
        if token in nltk.corpus.stopwords.words('english'):
            continue

        if not token.isalpha():
            continue

        if len(token) < 2:
            continue

        token = lemmatizer.lemmatize(token)
        clean_tokens.append(token)

    clean_string = ' '.join(clean_tokens)

    return clean_string


def save_text(file_properties, clean_text):
    """ Saves extracted text to JSON file.

    Args:
        file_properties(FileProperties): Object containing text output path.
        clean_text(str): Cleaned text string.

    """
    output_dict = {
        'Clean Content': clean_text
    }

    output_json = json.dumps(output_dict, indent=4)
    output_file = open(file_properties.text_path, 'wt')
    output_file.write(output_json)
    output_file.close()


def extract_text(file):
    """ Extracts text from file using Tika Server.

    Args:
        file: File bytes object.

    Returns:
        dict: Tika response dictionary.

    """
    tika_response = requests.put(
        url=f'http://127.0.0.1:80/tika',
        data=file,
        headers={
            'X-Tika-PDFOcrStrategy': 'no_ocr',
            'X-Tika-OCRLanguage': 'eng',
            'X-Tika-OCRTimeout': '1500',
            'Accept': 'text/plain'
        },
        timeout=1500
    )

    tika_response = {
        'content': tika_response.text,
        'status': tika_response.status_code
    }

    return tika_response
