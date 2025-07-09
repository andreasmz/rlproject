from andreas2048 import *

game = Game()
game.grid = np.full(shape=(4,4), fill_value=0)
for i in range(game.grid.shape[0]):
    for j in range(game.grid.shape[1]):
        game.grid[i, j] = i*4+j
#game.grid[0,0] = 1

import time

grid_r = game.grid.copy()

a = [1,2,3,4]

t0 = time.perf_counter()


    #game.origin_table[*grid_r[r]]
t1 = time.perf_counter()

print(t1-t0)

# print(game)
# print(game.get_moves())
# print(game.try_spawn())
# print(game.try_move(Action.RIGHT, no_spawn=True))
# print(game.try_move(Action.RIGHT, no_spawn=True))
# print(game.try_move(Action.RIGHT, no_spawn=True))
# print(game)