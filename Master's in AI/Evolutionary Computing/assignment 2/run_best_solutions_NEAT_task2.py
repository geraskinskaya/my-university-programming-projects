"""
Created on Wed Sep 29 22:05:08 2021

@author: doist
"""

# imports framework
import sys, os

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from NEAT_controller import player_controller
from controller import Controller

# imports other libs
import numpy as np
import pickle
import neat

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'NEAT_task2_e125_pop10_gen100'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  multiplemode="yes",
                  enemies=[1,2,3,4,5,6,7,8],
                  player_controller=player_controller(),
                  enemymode="static",
                  enemy_controller=Controller(),
                  level=2,
                  speed="fastest")

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-neat-task2.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


mean_fitness_per_run = []
mean_gain_per_run = []
for i in range(10):    
    sol_genome = pickle.load(open('winner_feedforward_' + str(experiment_name) + '_run_' + str(i), 'rb'))
    sol_net = neat.nn.FeedForwardNetwork.create(sol_genome, config)\
    
    fitnesses = []
    player_lives = []
    enemy_lives = []
    times = []
    gains = []
    
    for j in range(5):
        fitness = []
        player_live = []
        enemy_live = []
        time = []
        gain = []
        for en in env.enemies:
            f,p,e,t = env.run_single(en, pcont=sol_net, econt=env.enemy_controller)
            fitness.append(f)
            player_live.append(p)
            enemy_live.append(e)
            time.append(t)
            gain.append(p-e)
        fitnesses.append(sum(fitness))
        player_lives.append(sum(player_live))
        enemy_lives.append(sum(enemy_live))
        times.append(sum(time))
        gains.append(sum(gain))
        
    mean_fitness = sum(fitnesses) / len(fitnesses)
    mean_player_lives = sum(player_lives) / len(player_lives)
    mean_enemy_lives = sum(enemy_lives) / len(enemy_lives)
    mean_times = sum(times) / len(times)
    mean_gain = sum(gains) / len(gains)
    
    mean_fitness_per_run.append(mean_fitness)
    mean_gain_per_run.append(mean_gain)
    
mean_fitness_per_run = np.array(mean_fitness_per_run)
mean_gain_per_run = np.array(mean_gain_per_run)

np.savetxt(experiment_name+'/mean_fitness_over_runs.csv', mean_fitness_per_run, delimiter=",")
np.savetxt(experiment_name+'/mean_gain_over_runs.csv', mean_gain_per_run, delimiter=",")


