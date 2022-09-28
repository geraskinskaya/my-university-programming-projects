# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 22:05:08 2021

@author: doist
"""

# imports framework
import sys, os

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from NEAT_controller import player_controller

# imports other libs
import numpy as np
import pickle
import neat

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'NEAT1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")


# tests saved demo solutions for each enemy
enemies = [1]

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-neat.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

for en in enemies:
    # Update the enemy
    env.update_parameter('enemies', [en])
    mean_fitness_per_run = []
    mean_gain_per_run = []
    # Load specialist controller
    for i in range(10):
        gain_run = []
        fitness_run = []
        for j in range(5):
            sol_genome = pickle.load(open('winner_feedforward_' + str(experiment_name) + '_run_' + str(i), 'rb'))
            sol_net = neat.nn.FeedForwardNetwork.create(sol_genome, config)
            print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
            f, p, e, t = env.play(pcont=sol_net)
            gain_run.append(p-e)
            fitness_run.append(f)

        mean_fitness = sum(fitness_run) / len(fitness_run)
        mean_gain = sum(gain_run) / len(gain_run)

        mean_fitness_per_run.append(mean_fitness)
        mean_gain_per_run.append(mean_gain)

    mean_fitness_per_run = np.array(mean_fitness_per_run)
    mean_gain_per_run = np.array(mean_gain_per_run)

    np.savetxt(experiment_name+f'/mean_fitness_over_runs_e{en}.csv', mean_fitness_per_run, delimiter=",")
    np.savetxt(experiment_name+f'/mean_gain_over_runs_e{en}.csv', mean_gain_per_run, delimiter=",")


