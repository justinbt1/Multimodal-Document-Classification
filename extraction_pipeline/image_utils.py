import os
import math
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
font = ImageFont.truetype("arial.ttf", 16, encoding="unic")


def create_dir(file_properties):
    """ Creates directory to output images to.

    Args:
        file_properties(FileProperties): Tracks extraction parameters for a file.

    """
    if not os.path.isdir(file_properties.image_path):
        os.makedirs(file_properties.image_path)


def save_pdf_image(file_properties, index, image):
    """ Saves Wand images to disk.

    Args:
        file_properties(FileProperties): Tracks extraction parameters for a file.
        index(int): Page index number.
        image: PDF page image.

    Returns:
        str: Output path for image.

    """

    output_path = f'{os.path.join(file_properties.image_path, str(index))}.jpeg'
    image.save(output_path, format='jpeg')

    return output_path


def transform_and_save(image, file_properties, page_count=0):
    """ Breaks images into pages, resizes and saves output to disk.

    Args:
        image: PIL image object.
        file_properties(FileProperties): Tracks extraction parameters for a file.
        page_count(int): Count of pages already processed for image.

    Returns:
        int: Count of output page images.

    """
    create_dir(file_properties)

    if image.mode != 'RGB':
        image = image.convert('RGB')
  
    width, height = image.size
    portrait = width < height

    if portrait:
        aspect = height / width
    else:
        aspect = width / height

    page_count = page_count

    if aspect <= 2:
        image = image.resize((500, 500))
        page_count += 1
        output_path = f'{os.path.join(file_properties.image_path, str(page_count))}.jpeg'
        image.save(output_path, 'JPEG', quality=100)
        return page_count

    if portrait:
        page_width = width
        page_height = round(width * 1.4142)
        pages = math.ceil(height / page_height)
    else:
        page_width = round(height * 1.4142)
        page_height = height
        pages = math.ceil(width / page_width)

    top = 0
    bottom = page_height
    left = 0
    right = page_width

    for page in range(pages):
        page_count += 1

        if page_count > 10:
            break

        cropped_image = image.crop((left, top, right, bottom))
        cropped_image = cropped_image.resize((500, 500))
        output_path = f'{os.path.join(file_properties.image_path, str(page_count))}.jpeg'
        cropped_image.save(output_path, 'JPEG', quality=100)

        if portrait:
            top = bottom
            bottom += page_height
        else:
            left = right
            right += page_width

    return page_count


def page_image(page_text, file_path):
    """ Converts page of text to a JPEG file.

    Args:
        page_text(str): Page text.
        file_path(str): Output filepath.

    """
    image = Image.new('L', (1748, 2480), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), page_text, font=font)
    image = image.resize((500, 500))
    image = image.convert('RGB')
    image.save(file_path, 'JPEG', quality=100)


def text_to_image(file_properties, file_text):
    """ Breaks text file into pages and converts to page JPEG images.

    Args:
        file_properties(FileProperties): Tracks extraction parameters for a file.
        file_text(str): Text extracted from file.

    """
    create_dir(file_properties)

    page = ''
    line_count = 0
    page_count = 0

    for line in file_text.split('\n'):
        page += f'{line}\n'
        line_count += 1

        if line_count == 126:
            page += f'{line}\n'
            page_count += 1
            page_image(page, os.path.join(file_properties.image_path, f'{page_count}.jpeg'))
            line_count = 0
            page = ''

        if page_count == 10:
            break

    if line_count and page_count < 10:
        page_count += 1
        page_image(page, os.path.join(file_properties.image_path, f'{page_count}.jpeg'))
