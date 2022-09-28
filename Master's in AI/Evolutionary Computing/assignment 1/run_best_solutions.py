# imports framework
import sys, os

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
#from demo_controller import player_controller
from NEAT_controller import player_controller

# imports other libs
import numpy as np

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'NEAT1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="normal",
                  enemymode="static",
                  level=2)

# tests saved demo solutions for each enemy
enemies = [1]

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
            sol = np.loadtxt('PSO_task1/best_e' + str(en) + '_run_'+ str(i)+'.txt')
            print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
            f, p, e, t = env.play(pcont=sol)
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


