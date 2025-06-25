import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from typing import Literal


class Action(Enum):
    UP = (0, "y", (-1, 0), range(0,4), range(0,4))
    DOWN = (1, "y", (1, 0),  range(3, -1, -1), range(0,4))
    LEFT = (2, "x", (0,-1), range(0,4), range(0,4))
    RIGHT = (3, "x", (0,1), range(0,4), range(3,-1,-1))
    
    axis: Literal["x", "y"]
    dy: int
    dx: int
    xrange: list[int]
    yrange: list[int]

    def __new__(cls, value, axis, dxy, yrange, xrange):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.axis = axis
        obj.dy = dxy[0]
        obj.dx = dxy[1]
        obj.yrange = [y for y in yrange]
        obj.xrange = [x for x in xrange]
        return obj

class Game:

    mpl_cmap = ListedColormap(["#414140", 
                       "#eee4da",  # 2
                       "#ede0c8", # 4
                       "#f2b179", # 8
                       "#f59563", # 16
                       "#f67c5f",  # 32
                       "#f65e3b", # 64
                       "#edcf73", # 128
                       "#edcc62", # 256
                       "#edc850", # 512
                       "#edc53f", # 1024
                       "#edc22d", # 2048
                       "#3c3a32", # >= 4096
                       ])
    mpl_norm = BoundaryNorm(range(0,13), ncolors=mpl_cmap.N)

    def __init__(self) -> None:
        self.grid = np.full(shape=(4,4), fill_value=0)
        self.score = 0
        self.alive = True
        self.try_spawn()
        self.try_spawn()
        
    def try_spawn(self) -> bool:
        if not self.alive:
            return False
        empty_fields = np.argwhere(self.grid == 0)
        if len(empty_fields) == 0:
            self.alive = False
            return False
        ij = empty_fields[np.random.randint(low=0, high=len(empty_fields))]
        x = np.random.choice([1,2], p=[0.8, 0.2])
        self.grid[ij[0], ij[1]] = x
        self.score += 2**x
        if len(self.get_moves()) == 0:
            self.alive = False
            return False
        return True

    def try_move(self, action: Action) -> bool:
        if not self.alive:
            return False
        moves = 0
        for i1, y1 in enumerate(action.yrange):
            for j1, x1 in enumerate(action.xrange):
                if self.grid[y1,x1] == 0:
                    continue
                if action.axis == "y":
                    y2 = None
                    for yy in action.yrange[:i1][::-1]: # Go as far up or down as possible until hitting another block
                        if self.grid[yy,x1] == 0:
                            y2 = yy
                            continue
                        elif self.grid[yy,x1] == self.grid[y1,x1]:
                            moves += 1
                            self.grid[yy,x1] = self.grid[yy,x1]+1
                            self.grid[y1,x1] = 0
                            break
                        else:
                            break
                    if y2 is not None:
                        moves += 1
                        self.grid[y2,x1] = self.grid[y1,x1]
                        self.grid[y1,x1] = 0        

                else:
                    x2 = None
                    for xx in action.xrange[:j1][::-1]: # Go as far up or down as possible until hitting another block
                        if self.grid[y1,xx] == 0:
                            x2 = xx
                            continue
                        elif self.grid[y1,xx] == self.grid[y1,x1]:
                            moves += 1
                            self.grid[y1,xx] = self.grid[y1,xx]+1
                            self.grid[y1,x1] = 0
                            break
                        else:
                            break
                    if x2 is not None:
                        moves += 1
                        self.grid[y1,x2] = self.grid[y1,x1]
                        self.grid[y1,x1] = 0
        if moves == 0:
            return False
        return self.try_spawn()
    
    def get_moves(self) -> list[Action]:
        all_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        actions = []
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                for a in all_actions[:]:
                    y2, x2 = y + a.dy, x + a.dx
                    if y2 < 0 or x2 < 0 or y2 >= self.grid.shape[0] or x2 >= self.grid.shape[1]:
                        continue
                    if self.grid[y2, x2] == 0 or self.grid[y2, x2] == self.grid[y, x]:
                        actions.append(a)
                        all_actions.remove(a)
                if len(all_actions) == 0:
                    break
        return actions

    
    def plot_on_axis(self, ax: Axes):
        ax.imshow(self.grid, cmap=Game.mpl_cmap, norm=Game.mpl_norm)
        ax.set_axis_off()
        if self.alive:
            ax.set_title(f"score: {self.score}")
        else:
            ax.set_title(f"Game over (score: {self.score})")

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y,x] != 0:
                    c = "white" if self.grid[y,x] >= 3 else "black"
                    fsize = 26 if self.grid[y,x] <= 6 else 20
                    plt.text(x, y, 2**self.grid[y,x], ha="center", va="center", color=c, fontsize=fsize)
