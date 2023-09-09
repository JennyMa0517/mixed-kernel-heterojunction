from os import path
from personalized_classification.utils import Common

data_path = path.join(path.dirname(__file__), "../", "data")

tool = Common(data_path)
train_data = tool.get_train_data()

print(train_data["x_train"].shape)
print(train_data["y_train"].shape)
print(train_data["x_test"].shape)
print(train_data["y_test"].shape)
