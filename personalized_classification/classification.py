import numpy as np
import logging
from numpy.typing import NDArray
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from personalized_classification.static import CLASSES
from personalized_classification.experimental_kernel import ExperimentalKernel
from typing import List


class Classification:
    def __init__(self, x_train: NDArray[np.int64], y_train: NDArray[np.int64],
                 x_test: NDArray[np.int64], y_test: NDArray[np.int64]) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __score(self, y_test: NDArray[np.int64],
                y_predict: NDArray[np.int64]) -> List[float]:
        predict_count = [0] * len(CLASSES)
        true_count = [0] * len(CLASSES)

        for i, value in enumerate(y_predict):
            true_count[y_test[i]] += 1
            if y_test[i] == value:
                predict_count[y_test[i]] += 1

        score = [predict_count[i] / true_count[i] for i in range(len(CLASSES))]

        return score

    def run(self) -> NDArray[np.float64]:
        result: List[List[float]] = []

        for mixing_ratio in [0.0, 0.25, 0.50, 0.75, 1.0]:
            logging.info(f"Classification mixing ratio: {mixing_ratio} begins")
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel=lambda x, y: ExperimentalKernel().mixed_kernel(
                    x, y, mixing_ratio)))

            logging.info("\tTraining...")
            clf.fit(self.x_train, self.y_train)
            logging.info("\tPredicting...")
            y_predict = np.asarray(clf.predict(self.x_test)).astype(np.int64)

            score = self.__score(self.y_test, y_predict)
            logging.info(f"\tScore: {score}")
            score.append(float(clf.score(self.x_test, self.y_test)))
            result.append(score)
            logging.info(
                f"Classification mixing ratio: {mixing_ratio} completes")

        return np.array(result)
