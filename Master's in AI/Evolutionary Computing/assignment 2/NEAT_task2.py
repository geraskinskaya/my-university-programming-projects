# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:06 2021

@author: doist
"""

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from NEAT_controller import player_controller
from controller import Controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

# import packages neat
import multiprocessing
import os
import pickle
import neat
#import visualize
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# define experiment name
experiment_name = 'NEAT_task2_e125_pop10_gen100'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes",
                  enemies=[1,2,5],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  enemy_controller=Controller(),
                  level=2,
                  speed="fastest")

# checks environment state
env.state_to_log() 

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    player_lives = []
    enemy_lives = []
    game_times = []

    # Run the given simulation for up to num_steps time steps.
    for e in env.enemies:
        f,p,e,t = env.run_single(e, pcont=net, econt=env.enemy_controller)

        fitnesses.append(f)
        player_lives.append(p)
        enemy_lives.append(e)
        game_times.append(t)
       
    # The genome's fitness is based on the gain.
    return sum(player_lives)-sum(enemy_lives)-sum(np.log10(game_times))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
    
    # Calculate mean and max fitness of each generation
    max_fitness = -1*num_enemies*100
    sum_fitness = 0.0
    for genome_id, genome in genomes:
        sum_fitness += genome.fitness
        if genome.fitness >= max_fitness:
            max_fitness = genome.fitness
    mean_fitness = sum_fitness / len(genomes)
    
    print('Maximum fitness: ',max_fitness)
    print('Mean fitness: ', mean_fitness)
    
    summary['experiment_run_{}'.format(experiment_run)]['Mean_fitness_per_generation'].append(mean_fitness)
    summary['experiment_run_{}'.format(experiment_run)]['Max_fitness_per_generation'].append(max_fitness)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat-task2.txt')
    print(config_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    pop = neat.Population(config)
    #pop = Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    #pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    
    # 2nd argument of .run specifies the number of generations of the NEAT algo. 
    #winner = pop.run(pe.evaluate, 100)
    winner = pop.run(eval_genomes, 100)

    # Save the winner.
    winner_name = 'winner_feedforward_{}_run_{}'.format(experiment_name, experiment_run)
    with open(winner_name, 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    num_of_repeats_experiment = 10
    global summary
    global num_enemies 
    num_enemies = len(env.enemies)
    summary = {}
    global experiment_run
    for i in range(num_of_repeats_experiment):
        experiment_run = i
        summary['experiment_run_{}'.format(experiment_run)] = {}
        summary['experiment_run_{}'.format(experiment_run)]['Mean_fitness_per_generation'] = []
        summary['experiment_run_{}'.format(experiment_run)]['Max_fitness_per_generation'] = []
        run()
    print(summary)

    means = []
    maxes = []
    for i in range(len(summary)):
        means.append(summary['experiment_run_{}'.format(i)]['Mean_fitness_per_generation'])
        maxes.append(summary['experiment_run_{}'.format(i)]['Max_fitness_per_generation'])
        
    print(means)
    means = np.array(means)
    export_file_name = 'result_means_{}.csv'.format(experiment_name)
    np.savetxt(export_file_name, means, delimiter=',')
    
    print(maxes)
    maxes = np.array(maxes)
    export_file_name = 'result_maxes_{}.csv'.format(experiment_name)
    np.savetxt(export_file_name, maxes, delimiter=',')
    