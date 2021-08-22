import io
from PIL import Image
from pdf2image import convert_from_bytes

# Project modules.
import text_utils
import image_utils

Image.MAX_IMAGE_PIXELS = None


def extract_image_text(image):
    """ OCR text extraction from image.

    Args:
        image: Page image.

    Returns:
        dict: Dictionary containing Tika response.

    """
    jpeg_file = io.BytesIO()

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(jpeg_file, dpi=(300, 300), format='JPEG')
    tika_response = text_utils.extract_text(jpeg_file.getvalue())

    return tika_response


def parse_pdf_file(file_properties):
    """ Extracts PDF texts, performs OCR, saves page images to disk.

    Args:
        file_properties: Tracks extraction parameters for a file.

    Returns:
        file_properties: Tracks extraction parameters for a file.

    """
    file = open(file_properties.file_path, 'rb')
    file_bytes = file.read()
    file.close()

    tika_response = text_utils.extract_text(file_bytes)
    page_images = convert_from_bytes(file_bytes, dpi=300, size=(500, 500))

    page_count = 0
    content = []

    for i, page in enumerate(page_images):
        if i < file_properties.n_page_images:
            page_count = image_utils.transform_and_save(page, file_properties, page_count)

        if not tika_response['content'].strip():
            response = extract_image_text(page)
            content.append(response['content'])
        else:
            if i >= file_properties.n_page_images:
                break

    file_properties.image_extracted = 1
    file_properties.tika_status = tika_response['status']

    if not tika_response['content'].strip():
        tika_response['content'] = ' '.join(content)
        file_properties.ocr = 1

    if tika_response['content']:
        file_properties.text_extracted = 1

    clean_text = text_utils.text_processing(tika_response['content'])
    text_utils.save_text(file_properties, clean_text)

    return file_properties


def parse_image_file(file_properties):
    """ Parse image file formats, extract text and page images.

    Args:
        file_properties: Tracks extraction parameters for a file.

    Returns:
        file_properties: Tracks extraction parameters for a file.

    """
    image = Image.open(file_properties.file_path)
    tiff_frames = image.n_frames

    if file_properties.ext in ('.tif', '.tiff'):
        if tiff_frames == 1:
            image_utils.transform_and_save(image, file_properties)
            tika_response = extract_image_text(image)
        else:
            page_count = 0

            tika_response = {
                'content': []
            }

            for i in range(tiff_frames):
                image.seek(i)

                if page_count < 10:
                    page_count = image_utils.transform_and_save(image, file_properties, page_count)

                response = extract_image_text(image)
                tika_response['status'] = response['status']
                tika_response['content'].append(response['content'])

            tika_response['content'] = ' '.join(tika_response['content'])
    else:
        image_utils.transform_and_save(image, file_properties)
        tika_response = extract_image_text(image)

    if tika_response['content']:
        clean_text = text_utils.text_processing(tika_response['content'])
        file_properties.text_extracted = 1
    else:
        clean_text = ''

    file_properties.tika_status = tika_response['status']
    file_properties.ocr = 1
    file_properties.image_extracted = 1

    text_utils.save_text(file_properties, clean_text)

    return file_properties


def parse_text_file(file_properties):
    """ Parse text file formats, extract text and generate page images.

    Args:
        file_properties: Tracks extraction parameters for a file.

    Returns:
        file_properties: Tracks extraction parameters for a file.

    """
    file = open(file_properties.file_path, 'rt')
    file_text = file.read()
    file.close()

    if file_text:
        clean_text = text_utils.text_processing(file_text)
        file_properties.text_extracted = 1
    else:
        clean_text = ''

    file_properties.tika_status = '0'
    text_utils.save_text(file_properties, clean_text)

    image_utils.text_to_image(file_properties, file_text)
    file_properties.image_extracted = 1

    return file_properties
