import csv
from os import walk, path
from typing import List


class Common:
    window_size = 160
    maximum_counting = 10000

    classes = ["N", "L", "R", "A", "V", "/"]
    n_classes = len(classes)
    count_classes = [0] * n_classes

    def __init__(self, data_path: str):
        self.data_path = data_path

    def read_data(self) -> List[List[int]]:
        x: List[List[int]] = []
        y: List[int] = []

        records: List[str] = []
        annotations: List[str] = []

        all_file_paths: List[str] = []
        for dirpath, dirnames, filenames in walk(self.data_path):
            for filename in filenames:
                full_path = path.join(dirpath, filename)
                all_file_paths.append(full_path)
        all_file_paths.sort()

        for file_path in all_file_paths:
            filename, file_extension = path.splitext(file_path)
            if file_extension == ".txt":
                annotations.append(file_path)
            elif file_extension == ".csv":
                records.append(file_path)

        for (record, annotation) in zip(records, annotations):
            signals: List[int] = []
            with open(record, "r") as csvfile:
                reader = csv.reader(
                    csvfile,
                    quotechar="|",
                    delimiter=",",
                )
                next(reader)  # skip header

                for row in reader:
                    signals.append(int(row[1]))

            beat = []
            with open(annotation, "r") as txtfile:
                content = txtfile.readlines()
                for index in range(1, len(content)):
                    splitted = list(
                        filter(lambda v: v != '',
                               content[index].strip().split(' ')))
                    pos = int(splitted[1])
                    arrhythmia_type = splitted[2]

                    if (arrhythmia_type in self.classes):
                        arrhythmia_index = self.classes.index(arrhythmia_type)
                        # avoid overfitting
                        if self.count_classes[
                                arrhythmia_index] > self.maximum_counting:
                            pass
                        else:
                            self.count_classes[arrhythmia_index] += 1
                            if self.window_size < pos < (len(signals) -
                                                         self.window_size):
                                beat = signals[pos - self.window_size + 1:pos +
                                               self.window_size]
                                x.append(beat)
                                y.append(arrhythmia_index)

        data: List[List[int]] = []
        for (i, v) in enumerate(x):
            data.append(v)
            data[i].append(y[i])

        return x
