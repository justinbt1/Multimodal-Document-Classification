import os
from collections import Counter


def index_files(parent_dir):
    labels = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            file_path = os.path.join(root, file)
            code = file_path.split('\\')[-2]
            labels.append(code)

    count_labels = Counter(labels)

    count_labels = {
        k: v for k, v in sorted(count_labels.items(), key=lambda item: item[1])
    }

    for cs8_code in count_labels:
        print(f'{cs8_code}: {count_labels[cs8_code]}')


if __name__ == '__main__':
    index_files(r'..\data\ndr_files')
