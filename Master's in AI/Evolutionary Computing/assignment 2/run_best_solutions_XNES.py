# imports framework
import sys, os

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
# from NEAT_controller import player_controller
from evoman.controller import Controller

# imports other libs
import numpy as np

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'XNES_e24_run_best'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
enemies = [1,2,3,4,5,6,7,8]

# initializes environment for multiple objective mode (generalist) with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                    enemies=enemies,
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest")


# # Update the enemy
# env.update_parameter('enemies', [en])

mean_fitness_per_run = []
mean_gain_per_run = []

for i in range(10):

    sol = np.loadtxt('xnes_2-4_uniform_init_test/best_run_'+ str(i)+'.txt')
    print('\n LOADING SAVED GENERALIST SOLUTION \n')

    gains = []
    fitnesses = []
    player_lives = []
    enemy_lives = []
    times = []
    
    for j in range(5):

        gain = []
        fitness = []
        player_life = []
        enemy_life = []
        time = []

        for en in env.enemies:

            f, p, e, t = env.run_single(en, pcont=sol, econt=env.enemy_controller)

            gain.append(p-e)
            fitness.append(f)
            player_life.append(p)
            enemy_life.append(e)
            time.append(t)

        gains.append(sum(gain))
        fitnesses.append(sum(fitness))
        player_lives.append(sum(player_life))
        enemy_lives.append(sum(enemy_life))
        times.append(sum(time))

    mean_gain = sum(gains) / len(gains)
    mean_fitness = sum(fitnesses) / len(fitnesses)
    mean_player_lives = sum(player_lives) / len(player_lives)
    mean_enemy_lives = sum(enemy_lives) / len(enemy_lives)
    mean_times = sum(times) / len(times)

    mean_fitness_per_run.append(mean_fitness)
    mean_gain_per_run.append(mean_gain)

mean_fitness_per_run = np.array(mean_fitness_per_run)
mean_gain_per_run = np.array(mean_gain_per_run)

np.savetxt(experiment_name+f'/mean_fitness_over_runs.csv', mean_fitness_per_run, delimiter=",")
np.savetxt(experiment_name+f'/mean_gain_over_runs.csv', mean_gain_per_run, delimiter=",")


