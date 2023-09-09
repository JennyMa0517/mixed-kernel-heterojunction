from os import path
from numpy import genfromtxt
from flask import Flask, request
from flask_cors import CORS
from typing import TypedDict, List


class DataPoint(TypedDict):
    x: List[float]
    y: List[float]
    z: List[float]
    c: List[float]


class MarkPoint(TypedDict):
    x: List[float]
    y: List[float]
    z: List[float]


class MinMax(TypedDict):
    min: float
    max: float


app = Flask(__name__)
CORS(app)

output_path = path.join(path.dirname(__file__), '../', 'output')


@app.route('/')
def read_mu() -> DataPoint:
    iter = request.args.get('iter')
    data = genfromtxt(
        'ei_iter25_ESSkernel/mu_3d_plot_iter_{}.csv'.format(iter),
        delimiter=',')
    return {
        'x': data[:, 0].tolist(),
        'y': data[:, 1].tolist(),
        'z': data[:, 2].tolist(),
        'c': data[:, 3].tolist()
    }


@app.route('/ei')
def read_ei() -> DataPoint:
    iter = request.args.get('iter')
    data = genfromtxt(
        'ei_iter25_ESSkernel/ei_3d_plot_iter_{}.csv'.format(iter),
        delimiter=',')
    return {
        'x': data[:, 0].tolist(),
        'y': data[:, 1].tolist(),
        'z': data[:, 2].tolist(),
        'c': data[:, 3].tolist()
    }


@app.route('/mark')
def read_mark() -> MarkPoint:
    iter = request.args.get('iter', 1)
    data = genfromtxt(
        'ei_iter25_ESSkernel/op_3d_plot_iter_{}.csv'.format(iter),
        delimiter=',')
    if int(iter) == 1:
        data = data.reshape((1, -1))

    return {
        'x': data[:, 0].tolist(),
        'y': data[:, 1].tolist(),
        'z': data[:, 2].tolist()
    }


@app.route('/minmax')
def read_min_max() -> MinMax:
    min = float('inf')
    max = float('-inf')
    for i in range(1, 25):
        d = genfromtxt('ei_iter25_ESSkernel/mu_3d_plot_iter_{}.csv'.format(i),
                       delimiter=',')
        c = d[:, 3]
        local_min = c.min()
        local_max = c.max()
        if local_min < min:
            min = local_min
        if local_max > max:
            max = local_max
    return {'min': min, 'max': max}


if __name__ == '__main__':
    app.run(port=5000)
