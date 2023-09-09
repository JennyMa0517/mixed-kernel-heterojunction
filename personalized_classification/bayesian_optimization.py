from os import path, makedirs
import numpy as np
import logging
from numpy.typing import NDArray, ArrayLike
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from personalized_classification.experimental_kernel import ExperimentalKernel
from typing import Tuple


class BayesianOptimization:
    random_state = 12345
    bounds = np.asarray([[-10, -6], [-8, -4], [0, 1]], dtype=int)
    iterations = 25
    init_samples = 3
    cv = 3
    lambdas = np.linspace(-10, -6, 25)
    gammas = np.linspace(-8, -4, 25)
    scales = np.linspace(0, 1, 25)

    def __init__(self, x_train: NDArray[np.int64], y_train: NDArray[np.int64],
                 x_test: NDArray[np.int64], y_test: NDArray[np.int64]) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __expected_improvement(self, x: NDArray[np.int64],
                               gaussian_process: gp.GaussianProcessRegressor,
                               sampled_loss: ArrayLike,
                               n_params: int) -> NDArray[np.float64]:
        x = x.reshape(-1, n_params)

        mu, sigma = gaussian_process.predict(x, return_std=True)
        loss_optimum = np.max(sampled_loss)

        with np.errstate(divide="ignore"):
            Z = (mu - loss_optimum) / sigma
            ei_result = (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei_result[sigma == 0.0] = 0.0

        ei_result *= -1.0
        return np.asarray(ei_result)

    def __sample_next(self, gaussian_process: gp.GaussianProcessRegressor,
                      sampled_loss: ArrayLike, bounds: NDArray[np.float64],
                      n_restarts: int) -> ArrayLike:
        best_sample_point = np.array([])
        min_loss_value = 1.0
        init_points = np.random.uniform(bounds[:, 0],
                                        bounds[:, 1],
                                        size=(n_restarts, bounds.shape[0]))

        for init in init_points:
            curr_min_result = minimize(fun=self.__expected_improvement,
                                       x0=init.reshape(1, -1),
                                       bounds=bounds,
                                       method='L-BFGS-B',
                                       args=(gaussian_process, sampled_loss,
                                             bounds.shape[0]))

            if curr_min_result.fun < min_loss_value:
                best_sample_point = curr_min_result.x
                min_loss_value = curr_min_result.fun

        return best_sample_point

    def __optimizer(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = []
        y = []
        init_points = np.random.uniform(self.bounds[:, 0],
                                        self.bounds[:, 1],
                                        size=(self.init_samples,
                                              self.bounds.shape[0]))

        for init in init_points:
            x.append(init)
            y.append(self.sample_loss(init))

        kernel = gp.kernels.ExpSineSquared()
        gp_model = gp.GaussianProcessRegressor(kernel=kernel,
                                               alpha=1e-5,
                                               n_restarts_optimizer=10,
                                               normalize_y=True)

        for _ in range(self.iterations):
            logging.info(f"Bayesian optimization: iteration {_} begins")
            # fit gaussian process regressor
            xp = np.array(x, dtype=float)
            yp = np.array(y, dtype=float)
            logging.info("\tfitting gaussian process...")
            gp_model.fit(xp, yp)

            logging.info("\tsampling next point...")
            # sample next point
            next_sample = np.asarray(
                self.__sample_next(
                    gaussian_process=gp_model,
                    sampled_loss=yp,
                    bounds=self.bounds,
                    n_restarts=100,
                ))

            # avoid duplicates
            if (np.abs(next_sample - xp) <= 1e-7).any():
                next_sample = np.random.uniform(self.bounds[:, 0],
                                                self.bounds[:, 1],
                                                size=self.bounds.shape[0])

            # update
            x.append(next_sample)
            y.append(self.sample_loss(next_sample))
            logging.info(f"Bayesian optimization: iteration {_} completes")

        xp = np.asarray(x, dtype=float)
        yp = np.asarray(y, dtype=float)
        return xp, yp

    def __save(
        self, sampled_params: NDArray[np.float64],
        sampled_loss: NDArray[np.float64], save_csv: bool, save_path: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64]]:
        kernel = gp.kernels.ExpSineSquared()
        gp_model = gp.GaussianProcessRegressor(kernel=kernel,
                                               alpha=1e-5,
                                               n_restarts_optimizer=10,
                                               normalize_y=True)
        mu_list = []  # median
        ei_list = []  # expected improvement
        op_list = []  # optimized next sampling points

        grid = np.array([[l, g, s] for l in self.lambdas for g in self.gammas
                         for s in self.scales])

        for i in range(3, sampled_params.shape[0] - 1):
            logging.info(f"Data save: {i} begins")
            gp_model.fit(X=sampled_params[:i + 1, :], y=sampled_loss[:i + 1])
            mu = np.asarray(gp_model.predict(grid))
            ei = -1.0 * self.__expected_improvement(
                x=grid,
                gaussian_process=gp_model,
                sampled_loss=sampled_loss[:i + 1],
                n_params=3)
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

    def sample_loss(self, params: NDArray[np.float64]) -> ArrayLike:
        logging.info(f"Sampling loss for params: {params}")
        return np.asarray(
            cross_val_score(SVC(
                random_state=self.random_state,
                kernel=lambda x, y: ExperimentalKernel().mixed_kernel(
                    x, y, params[0], params[1], params[2])),
                            X=self.x_train,
                            y=self.y_train,
                            cv=self.cv).mean())

    def run(
        self,
        save_csv: bool = True,
        save_path: str = path.join(path.dirname(__file__), "../", "output")
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64]]:
        # execute bayesian optimization
        sampled_params, sampled_loss = self.__optimizer()

        # save optimization results for plotting
        return self.__save(sampled_params, sampled_loss, save_csv, save_path)
