import os
import shutil


def move_files(origin, destination, cs8_codes):
    for root, dirs, files in os.walk(origin):
        for file in files:
            path = os.path.join(root, file)
            file_name = os.path.split(path)[-1]

            for cs8_code in cs8_codes:
                if cs8_code.upper() in file_name:
                    new_dir = f'{destination}\\{cs8_code}'

                    if not os.path.isdir(new_dir):
                        os.mkdir(new_dir)

                    new_path = f'{new_dir}\\{file_name}'

                    if os.path.exists(path):
                        shutil.move(path, new_path)


if __name__ == '__main__':

    origin_path = r'Original file path'
    destination_path = r'Destination file path'

    cs8_codes_list = [
        'geol_geow',
        'geol_sed',
        'gphys_gen',
        'log_sum',
        'pre_site',
        'vsp_file'
    ]

    move_files(origin_path, destination_path, cs8_codes_list)
