# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:21:40 2021

@author: doist
"""

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
#from demo_controller import player_controller
from NEAT_controller import player_controller

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

#from neat_population_adjusted import Population

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# define experiment name
experiment_name = 'NEAT4'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[4],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# checks environment state
env.state_to_log() 

# start of neat algorithm
runs_per_net = 1

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        # Run the given simulation for up to num_steps time steps.
        f,p,e,t = env.play(pcont=net)
        fitnesses.append(f)
        
    # saves simulation state
    #env.save_state() 
        
    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
    
    # Calculate mean and max fitness of each generation
    max_fitness = -100
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
    config_path = os.path.join(local_dir, 'config-neat.txt')
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
    
    # visualizations can be skipped
    '''
    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)        

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)
    '''

if __name__ == '__main__':
    num_of_repeats_experiment = 10
    global summary
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
    