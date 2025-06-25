from andreas2048 import *

game = Game()
game.grid = np.full(shape=(4,4), fill_value=0)
n = 2
for i in range(4):
    for j in range(4):
        game.grid[i, j] = n
        n += 1
game.grid[0,0] = 0 
print(game.try_move(Action.LEFT))
print(game.get_moves())
print(game.alive)
print(game.grid)