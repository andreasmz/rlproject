from andreas2048 import *

game = Game()
game.state.grid = np.full(shape=(4,4), fill_value=0, dtype=np.uint8)
for i in range(game.grid.shape[0]):
    for j in range(game.grid.shape[1]):
        game.grid[i, 3-j] = i*4+j

print(game)

print(game.next(Action.LEFT))