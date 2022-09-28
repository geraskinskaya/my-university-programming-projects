# imports framework
import sys

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

# experiment_name = 'multi_demo'
experiment_name = 'B_sigma_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes simulation in individual evolution mode, for multiple static enemies
def initalize_environment(n_hidden_neurons, enemies):
    global env
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    env.state_to_log()  # checks environment state



# ----------------------------------------------------------------------
# XNES ALGORITHM

import joblib
import random

import scipy as sp
from scipy import (dot, eye, randn, asarray, array, trace, log, exp, sqrt, mean, sum, argsort, square, arange)
from scipy.stats import multivariate_normal, norm
from scipy.linalg import (det, expm)


class XNES(object):
    def __init__(self, f, mu, amat,
                 eta_mu=1.0, eta_sigma=None, eta_bmat=None,
                 npop=None, use_fshape=True, use_adasam=False, patience=100, n_jobs=1):
        self.f = f
        self.mu = mu
        self.eta_mu = eta_mu
        self.use_adasam = use_adasam
        self.n_jobs = n_jobs

        dim = len(mu)
        sigma = abs(det(amat)) ** (1.0 / dim)
        bmat = amat * (1.0 / sigma)
        self.dim = dim
        self.sigma = sigma
        self.bmat = bmat

        # default population size and learning rates
        npop = int(4 + 3 * np.log(dim)) if npop is None else npop
        eta_sigma = 3 * (3 + np.log(dim)) * (1.0 / (5 * dim * np.sqrt(dim))) if eta_sigma is None else eta_sigma
        eta_bmat = 3 * (3 + np.log(dim)) * (1.0 / (5 * dim * np.sqrt(dim))) if eta_bmat is None else eta_bmat
        self.npop = npop
        self.eta_sigma = eta_sigma
        self.eta_bmat = eta_bmat

        # compute utilities if using fitness shaping
        if use_fshape:
            a = np.log(1 + 0.5 * npop)
            utilities = np.array([max(0, a - np.log(k)) for k in range(1, npop + 1)])
            utilities /= np.sum(utilities)
            utilities -= 1.0 / npop  # broadcast
            utilities = utilities[::-1]  # ascending order
        else:
            utilities = None
        self.use_fshape = use_fshape
        self.utilities = utilities

        # stuff for adasam
        self.eta_sigma_init = eta_sigma
        self.sigma_old = None

        # logging
        self.fitness_best = -1000
        self.mu_best = None
        self.done = False
        self.counter = 0
        self.patience = patience
        self.history = {'eta_sigma': [], 'sigma': [], 'fitness_mean': [], 'fitness_max': []}

        # do not use these when hill-climbing
        if npop == 1:
            self.use_fshape = False
            self.use_adasam = False

    def step(self, niter):
        """ xNES """
        f = self.f
        mu, sigma, bmat = self.mu, self.sigma, self.bmat
        eta_mu, eta_sigma, eta_bmat = self.eta_mu, self.eta_sigma, self.eta_bmat
        npop = self.npop
        dim = self.dim
        sigma_old = self.sigma_old

        eyemat = np.eye(dim)

        with joblib.Parallel(n_jobs=1) as parallel:

            for i in range(niter):
                print('ITERATION:',i)
                s_try = np.random.randn(npop, dim)
                z_try = mu + sigma * np.dot(s_try, bmat)  # broadcast

                f_try = parallel(joblib.delayed(f)(z) for z in z_try)
                f_try = np.asarray(f_try)

                # save if best
                fitness_mean = np.mean(f_try)
                fitness_max = np.max(f_try)
                print(f'MEAN: {fitness_mean}, MAX:{fitness_max}')
                if fitness_max > self.fitness_best:
                    self.fitness_best = fitness_max
                    self.mu_best = mu.copy()
                    self.counter = 0
                else:
                    self.counter += 1
                if self.counter > self.patience:
                    print('----------------PARAMETER RESET--------------------')
                    n_vars = self.dim
                    amat = np.eye(n_vars)
                    self.sigma = abs(det(amat)) ** (1.0 / dim)
                    self.bmat = amat * (1.0 / sigma)


                isort = np.argsort(f_try)
                f_try = f_try[isort]
                s_try = s_try[isort]
                z_try = z_try[isort]

                u_try = self.utilities if self.use_fshape else f_try

                if self.use_adasam and sigma_old is not None:  # sigma_old must be available
                    eta_sigma = self.adasam(eta_sigma, mu, sigma, bmat, sigma_old, z_try)

                dj_delta = np.dot(u_try, s_try)
                dj_mmat = np.dot(s_try.T, s_try * u_try.reshape(npop, 1)) - np.sum(u_try) * eyemat
                dj_sigma = np.trace(dj_mmat) * (1.0 / dim)
                dj_bmat = dj_mmat - dj_sigma * eyemat

                sigma_old = sigma

                # update
                mu += eta_mu * sigma * np.dot(bmat, dj_delta)
                sigma *= np.exp(0.5 * eta_sigma * dj_sigma)
                bmat = np.dot(bmat, expm(0.5 * eta_bmat * dj_bmat))

                # logging
                self.history['fitness_mean'].append(fitness_mean)
                self.history['fitness_max'].append(fitness_max)
                self.history['sigma'].append(sigma)
                self.history['eta_sigma'].append(eta_sigma)

        # keep last results
        self.mu, self.sigma, self.bmat = mu, sigma, bmat
        self.eta_sigma = eta_sigma
        self.sigma_old = sigma_old

    def adasam(self, eta_sigma, mu, sigma, bmat, sigma_old, z_try):
        """ Adaptation sampling """
        eta_sigma_init = self.eta_sigma_init
        dim = self.dim
        c = .1
        rho = 0.5 - 1. / (3 * (dim + 1))  # empirical

        bbmat = dot(bmat.T, bmat)
        cov = sigma ** 2 * bbmat
        sigma_ = sigma * sqrt(sigma * (1. / sigma_old))  # increase by 1.5
        cov_ = sigma_ ** 2 * bbmat

        p0 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov)
        p1 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov_)
        w = exp(p1 - p0)

        # Mann-Whitney. It is assumed z_try was in ascending order.
        n = self.npop
        n_ = sum(w)
        u_ = sum(w * (arange(n) + 0.5))

        u_mu = n * n_ * 0.5
        u_sigma = sqrt(n * n_ * (n + n_ + 1) / 12.)
        cum = norm.cdf(u_, loc=u_mu, scale=u_sigma)

        if cum < rho:
            return (1 - c) * eta_sigma + c * eta_sigma_init
        else:
            return min(1, (1 + c) * eta_sigma)


# -----------------------------------------------------------------------

def initialize_weights(n_hidden, n_sensors):
    # Hacky method of a Xavier initalization
    # Evoman slices the full weight array according to indices, we initialize based on these indices
    bias1 = np.zeros(n_hidden)
    weights1 = np.random.normal(0, 1/n_sensors, n_sensors*n_hidden)
    bias2 = np.zeros(5)
    weights2 = np.random.normal(0, 1/n_hidden, n_hidden*5)

    weights = np.hstack((bias1, weights1, bias2, weights2))

    return weights

def f(x):
    fitness = 0
    for e in env.enemies:
        f,p,e,t = env.run_single(e, pcont=x, econt=env.enemy_controller)
        fitness += (p - e + np.log10(t))
    print(fitness)
    return fitness

def run():
    ini = time.time()  # sets time marker

    # Magic numbers
    enemies = [1, 2, 5]
    n_hidden_neurons = 10
    initalize_environment(n_hidden_neurons, enemies)
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    num_runs = 10
    num_iterations = 100
    population_size = None

    mean_over_runs = []
    max_over_runs = []

    for i in range(num_runs):
        print('RUN:', i + 1)

        global mean_over_iterations
        mean_over_iterations = []
        global max_over_iterations
        max_over_iterations = []

        # mu = initialize_weights(10, env.get_num_sensors())
        mu = np.random.uniform(low=-1, high=1, size=(n_vars,))
        amat = np.eye(n_vars)

        global xnes
        xnes = XNES(f, mu, amat, npop=population_size, use_adasam=False, eta_bmat=None, eta_sigma=None, patience=15)
        xnes.step(num_iterations)

        print(xnes.sigma)
        print(xnes.bmat)

        print("Current: ({},{})".format(*xnes.mu))
        np.savetxt(experiment_name + f'/best_run_{i}.txt', xnes.mu_best)
        mean_over_runs.append(xnes.history['fitness_mean'])
        max_over_runs.append(xnes.history['fitness_max'])

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(3,1)
        # axs[0].plot(xnes.history['fitness_max'])
        # axs[1].plot(xnes.history['fitness_mean'])
        # axs[0].set_ylabel('fitness_max')
        # axs[1].set_ylabel('fitness_mean')
        # fig.show()

    mean_over_runs = np.array(mean_over_runs)
    max_over_runs = np.array(max_over_runs)

    np.savetxt(experiment_name + f'/mean_over_runs.csv', mean_over_runs, delimiter=",")
    np.savetxt(experiment_name + f'/max_over_runs.csv', max_over_runs, delimiter=",")

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

    print('end')


if __name__ == '__main__':
    run()
