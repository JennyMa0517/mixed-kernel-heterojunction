from os import path
from personalized_classification.utils import Common

data_path = path.join(path.dirname(__file__), "../", "data")

tool = Common(data_path)
data = tool.read_data()
