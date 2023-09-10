from os import path, makedirs
import numpy as np
import logging
import random
from numpy.typing import NDArray, ArrayLike
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from personalized_classification.experimental_kernel import ExperimentalKernel
from typing import Tuple


class BayesianOptimization:
    opt_bound_1 = [-10, -6]
    opt_bound_2 = [-8, -4]
    opt_bound_3 = [0, 1]

    random_state = random.randint(0, 10000)
    iterations = 25
    init_samples = 3
    cv = 3
    method = 'L-BFGS-B'
    alpha = 1e-5
    n_restarts_optimizer = 10
    n_restarts_sample = 100
    epsilon = 1e-7

    opt_bounds = np.asarray([opt_bound_1, opt_bound_2, opt_bound_3], dtype=int)
    lambdas = np.linspace(opt_bound_1[0], opt_bound_1[1], iterations)
    gammas = np.linspace(opt_bound_2[0], opt_bound_2[1], iterations)
    scales = np.linspace(opt_bound_3[0], opt_bound_3[1], iterations)

    def __init__(self, x_train: NDArray[np.int64], y_train: NDArray[np.int64],
                 x_test: NDArray[np.int64], y_test: NDArray[np.int64]) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __expected_improvement(self, x: NDArray[np.float64],
                               gaussian_process: gp.GaussianProcessRegressor,
                               sampled_loss: ArrayLike,
                               init_samples: int) -> NDArray[np.float64]:
        max_loss = np.max(sampled_loss)
        mu, sigma = gaussian_process.predict(x.reshape(-1, init_samples),
                                             return_std=True)
        phi = mu - max_loss

        with np.errstate(divide="ignore"):
            z = phi / sigma

        ei_result = sigma * norm.pdf(z) + phi * norm.cdf(z)
        ei_result[sigma == 0.0] = 0.0
        ei_result *= -1.0

        return np.asarray(ei_result)

    def __sample_next(self, gaussian_process: gp.GaussianProcessRegressor,
                      sampled_loss: ArrayLike, bounds: NDArray[np.float64],
                      n_restarts: int) -> ArrayLike:
        best_sample_point = np.array([])
        min_loss_value = 1.0
        initial_points = np.random.uniform(low=bounds[:, 0],
                                           high=bounds[:, 1],
                                           size=(n_restarts, bounds.shape[0]))

        for p in initial_points:
            min_result = minimize(
                fun=self.__expected_improvement,
                x0=(p.reshape(1, -1)[0]).tolist(),
                args=(gaussian_process, sampled_loss, bounds.shape[0]),
                bounds=bounds,
                method=self.method,
            )

            if min_result.fun < min_loss_value:
                best_sample_point = min_result.x
                min_loss_value = min_result.fun

        return best_sample_point

    def __sample_loss(self, params: NDArray[np.float64]) -> ArrayLike:
        logging.info(f"Sampling loss for params: {params}")

        def kernel_func(x: NDArray[np.int64],
                        y: NDArray[np.int64]) -> NDArray[np.float64]:
            return ExperimentalKernel().mixed_kernel(x, y, params[2],
                                                     params[0], params[1])

        return np.asarray(
            cross_val_score(SVC(random_state=self.random_state,
                                kernel=kernel_func),
                            X=self.x_train,
                            y=self.y_train,
                            cv=self.cv).mean())

    def optimizer(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        initial_points = np.random.uniform(low=self.opt_bounds[:, 0],
                                           high=self.opt_bounds[:, 1],
                                           size=(self.init_samples,
                                                 self.opt_bounds.shape[0]))

        x = []
        y = []
        for p in initial_points:
            x.append(p)
            y.append(self.__sample_loss(p))

        model = gp.GaussianProcessRegressor(
            kernel=gp.kernels.ExpSineSquared(),
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True)

        for _ in range(self.iterations):
            logging.info(f"Bayesian optimization: iteration {_} begins")
            # fit gaussian process regressor
            sampled_params = np.array(x, dtype=float)
            sampled_loss = np.array(y, dtype=float)
            logging.info("\tfitting gaussian process...")
            model.fit(sampled_params, sampled_loss)

            logging.info("\tsampling next point...")
            # sample next point
            next_sample = np.asarray(
                self.__sample_next(
                    gaussian_process=model,
                    sampled_loss=sampled_loss,
                    bounds=self.opt_bounds,
                    n_restarts=self.n_restarts_sample,
                ))

            # avoid duplicates
            if (np.abs(next_sample - sampled_params) <= self.epsilon).any():
                next_sample = np.random.uniform(low=self.opt_bounds[:, 0],
                                                high=self.opt_bounds[:, 1],
                                                size=self.opt_bounds.shape[0])
            next_sample_loss = self.__sample_loss(next_sample)

            # update
            x.append(next_sample)
            y.append(next_sample_loss)
            logging.info(f"Bayesian optimization: iteration {_} completes")

        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def save(
        self, sampled_params: NDArray[np.float64],
        sampled_loss: NDArray[np.float64], save_csv: bool, save_path: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64]]:
        model = gp.GaussianProcessRegressor(
            kernel=gp.kernels.ExpSineSquared(),
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True)
        grid = np.array([[lambdaa, gamma, scale] for lambdaa in self.lambdas
                         for gamma in self.gammas for scale in self.scales])

        mu_list = []  # median
        ei_list = []  # expected improvement
        op_list = []  # optimized next sampling points
        for i in range(self.init_samples, sampled_params.shape[0] - 1):
            logging.info(f"Data save: {i} begins")
            model.fit(X=sampled_params[:i + 1, :], y=sampled_loss[:i + 1])
            mu = np.asarray(model.predict(grid))
            ei = -1.0 * self.__expected_improvement(
                x=grid,
                gaussian_process=model,
                sampled_loss=sampled_loss[:i + 1],
                init_samples=self.init_samples)
            op = sampled_params[i + 1, [0, 1, 2]]

            mu_list.append(mu)
            ei_list.append(ei)
            op_list.append(op)
            logging.info(f"Data save: {i} completes")

        if save_csv:
            logging.info("Saving data as csv...")
            if not path.exists(save_path):
                logging.info(f"Creating directory: {save_path}")
                makedirs(save_path)

            for j, v in enumerate(ei_list):
                np.savetxt(path.join(save_path, f'ei_3d_plot_iter_{j+1}.csv'),
                           np.append(grid, v.reshape((-1, 1)), axis=1),
                           delimiter=",")
            for j, v in enumerate(mu_list):
                np.savetxt(path.join(save_path, f'mu_3d_plot_iter_{j+1}.csv'),
                           np.append(grid, v.reshape((-1, 1)), axis=1),
                           delimiter=",")
            for j, _ in enumerate(op_list):
                np.savetxt(path.join(save_path, f'op_3d_plot_iter_{j+1}.csv'),
                           op_list[:j + 1],
                           delimiter=",")
            logging.info("Data saved as csv")

        return grid, np.asarray(ei_list, dtype=float), np.asarray(
            mu_list, dtype=float), np.asarray(op_list, dtype=float)

    def run(
        self,
        save_csv: bool = True,
        save_path: str = path.join(path.dirname(__file__), "../", "output")
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64]]:
        # execute bayesian optimization
        sampled_params, sampled_loss = self.optimizer()

        # save optimization results for plotting
        return self.save(sampled_params, sampled_loss, save_csv, save_path)
