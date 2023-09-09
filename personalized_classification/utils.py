from os import walk, path
from typing import List


class Common:
    window_size = 160
    maximum_counting = 10000

    classes = ["N", "L", "R", "A", "V", "/"]
    n_classes = len(classes)
    count_classes = [0] * n_classes

    x: List[str] = []
    y: List[str] = []

    records: List[str] = []
    annotations: List[str] = []

    def __init__(self, data_path: str):
        self.data_path = data_path

    def read_files(self) -> None:
        all_file_paths: List[str] = []
        for dirpath, dirnames, filenames in walk(self.data_path):
            for filename in filenames:
                full_path = path.join(dirpath, filename)
                all_file_paths.append(full_path)

        for file_path in all_file_paths:
            filename, file_extension = path.splitext(file_path)
            if file_extension == ".txt":
                self.annotations.append(file_path)
            elif file_extension == ".csv":
                self.records.append(file_path)

        for record in self.records:
            sigals = []
