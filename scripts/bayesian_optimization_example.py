import logging
from os import path
from personalized_classification.utils import Utils
from personalized_classification.bayesian_optimization import BayesianOptimization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])

data_path = path.join(path.dirname(__file__), "../", "data")
output_path = path.join(path.dirname(__file__), "../", "output")

utils = Utils(data_path)
train_data = utils.get_train_data()

result = BayesianOptimization(train_data["x_train"], train_data["y_train"],
                              train_data["x_test"],
                              train_data["y_test"]).run(True, output_path)

print(result)
