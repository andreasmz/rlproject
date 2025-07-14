from andreas2048 import *

game = Game()
game.grid = np.full(shape=(4,4), fill_value=0)
for i in range(game.grid.shape[0]):
    for j in range(game.grid.shape[1]):
        game.grid[i, j] = 0
game.grid[:,3] = [2,2,1,1]
print(game.get_moves())
print(game.try_move(Action.DOWN))
print(game.get_moves())
print(game.alive)
print(game.grid)