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


sol = np.loadtxt('xnes_1-2-5_uniform_init_jaimie/best_run_'+ str(5)+'.txt')
print('\n LOADING SAVED GENERALIST SOLUTION \n')

gains = []
fitnesses = []
player_lives = []
enemy_lives = []
times = []

for j in range(5):

    player_life = []
    enemy_life = []


    for en in env.enemies:

        f, p, e, t = env.run_single(en, pcont=sol, econt=env.enemy_controller)

        player_life.append(p)
        enemy_life.append(e)

    player_lives.append(player_life)
    enemy_lives.append(enemy_life)

player_lives = np.array(player_lives)
enemy_lives = np.array(enemy_lives)

player_mean = np.mean(player_lives, axis=0)
enemy_mean = np.mean(enemy_lives, axis=0)


print(player_mean)
print(enemy_mean)


