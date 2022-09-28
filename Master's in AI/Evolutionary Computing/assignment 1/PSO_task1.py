# imports framework
import sys
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import time
import numpy as np
from math import fabs, sqrt
import glob, os
import matplotlib.pyplot as plt



# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'PSO_task1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes simulation in individual evolution mode, for single static enemy
def initalize_environment(n_hidden_neurons, enemy):    
    global env
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    env.state_to_log()  # checks environment state


### Particle Swarm Optimization Algorithm ###

def simulation(x):

    f, p, e, t = env.play(pcont=x)

    return -1 * f  # PSO tries to find a minimum cost, we need to find maximum fitness

# def check_improvement(fitness, best):
#
#     if  best <= fitness:
#
#         # print('best <= fitness')
#         # print(best)
#         # print(fitness)
#
#         notimproved += 1
#
#     else:
#
#         print('best > fitness')
#         print(best)
#         print(fitness)
#         print(weights)
#
#         best = fitness
#         best_weights = weights
#         notimproved = 0
#
#     if notimproved >= 15:
#         notimproved = 0
#         optimizer.reset()
#         print('RESET EXECUTED')
#
#     print('NOT IMPROVED:', notimproved)



def f(x):

    n_particles = x.shape[0]
    j = [simulation(x[i]) for i in range(n_particles)]
    j = np.array(j)
    mean = np.mean(j*-1) # calculate variable once for efficiency
    max = np.max(j*-1)
    print('ITERATION MEAN FITNESS:', mean)
    # reconvert to actual fitness for visualization

    max_over_iterations.append(max)
    mean_over_iterations.append(mean)

    return j


def run():

    ini = time.time()  # sets time marker

    # Magic numbers
    n_hidden_neurons = 10
    weight_init_bounds = [-1, 1]
    num_runs = 10
    n_particles = 20 
    num_iterations = 100
    # swarm_params = {'c1': 1.49, 'c2': 1.49, 'w': 0.7298} # 2 0.5 0.7298
    enemy = 6
    start_opts = {'c1':2.5, 'c2':0.5, 'w':0.7298}
    # end_opts= {'c1':0.5, 'c2':2.5, 'w':0.4}     # Ref:[1]
    oh_strategy={ "w":'exp_decay', "c1":'nonlin_mod',"c2":'lin_variation'}

    initalize_environment(n_hidden_neurons, enemy)

    # Call instance of PSO
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    minimum = np.ones(n_vars) * weight_init_bounds[0]
    maximum = np.ones(n_vars) * weight_init_bounds[1]

    mean_over_runs = []
    max_over_runs = []

    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    file_aux.close()


    for i in range(num_runs):

        global mean_over_iterations
        mean_over_iterations = []
        global max_over_iterations
        max_over_iterations = []

        notimproved = 0

        print('RUN:', i+1)

        global optimizer
        # optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, bounds=(minimum, maximum), dimensions=n_vars, options=swarm_params)
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                            bounds=(minimum, maximum),
                                            dimensions=n_vars,
                                            oh_strategy=oh_strategy,
                                            options=start_opts,
                                            bh_strategy='nearest')

        fitness, weights = optimizer.optimize(f, iters=num_iterations, verbose=True)

        np.savetxt(experiment_name+f'/best_e{enemy}_run_{i}.txt', weights)
        # fitness = fitness * -1
        #
        # if fitness > best:
        #     best = fitness
        #     best_weights = weights
        #
        # # print('test')
        # # print(fitness, weights)

        mean_over_runs.append(mean_over_iterations)
        max_over_runs.append(max_over_iterations)



    mean_over_runs = np.array(mean_over_runs)
    max_over_runs = np.array(max_over_runs)

    np.savetxt(experiment_name+f'/mean_over_runs_e{enemy}.csv', mean_over_runs, delimiter=",")
    np.savetxt(experiment_name+f'/max_over_runs_e{enemy}.csv', max_over_runs, delimiter=",")

    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


    # file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    # file.close()
    #
    # env.state_to_log() # checks environment state

    print('end')


if __name__ == '__main__':
    run()
