from andreas2048 import *

game = Game()
game.grid = np.full(shape=(4,4), fill_value=0)
for i in range(game.grid.shape[0]):
    for j in range(game.grid.shape[1]):
        game.grid[i, j] = 0
game.grid[3,0] = 1
game.grid[3,1] = 1
game.grid[3,2] = 2
game.grid[3,3] = 3
game.grid[2,2] = 1
print(game.try_move(Action.DOWN))
print(game.get_moves())
print(game.alive)
print(game.grid)