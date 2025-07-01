import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Arrow, Rectangle
from typing import Literal, Callable


class Action(Enum):
    UP = (0, "y", (-1, 0))
    DOWN = (1, "y", (1, 0))
    LEFT = (2, "x", (0,-1))
    RIGHT = (3, "x", (0,1))
    
    axis: Literal["x", "y"]
    dy: int
    dx: int
    xrange: Callable[[int], list[int]]
    yrange: Callable[[int], list[int]]

    def __new__(cls, value, axis, dxy):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.axis = axis
        obj.dy = dxy[0]
        obj.dx = dxy[1]
        obj.yrange = lambda s: [y for y in (range(0,s) if dxy[0] < 0 else range(s-1,-1,-1))]
        obj.xrange = lambda s: [y for y in (range(0,s) if dxy[1] < 0 else range(s-1,-1,-1))]
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

    def __init__(self, shape:tuple[int, int] = (4,4), seed: int|None = None, save_random_state: bool = False) -> None:
        self.grid = np.zeros(shape=shape, dtype=np.uint8)
        self.save_random_state = save_random_state
        self.score = 0
        self._alive = True
        self.rnd = np.random.default_rng(seed=seed)
        self.history: list[np.ndarray] = [np.stack((self.grid, np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)), axis=0)]
        self.score_history = [self.score]
        self._rnd_history = [self.rnd.bit_generator.state]

        self.xyt_to_idx = lambda y, x, t: y*self.grid.shape[0] + x + t*self.grid.size + 2
        self.idx_to_xyt = lambda idx: ( (int((idx-2) % self.grid.size) // self.grid.shape[0]), int((idx-2) % self.grid.shape[0]) , int((idx-2) // self.grid.size) )

        self.try_spawn()
        self.try_spawn()

    @property
    def move_count(self) -> int:
        return len(self.history) - 1
    
    @property
    def alive(self) -> bool:
        return self._alive
    
    @property
    def highest_tile(self) -> int:
        return 2**int(np.max(self.grid))
    
    @property
    def reward(self) -> int:
        return self.score - self.score_history[-1]
        
    def try_spawn(self) -> bool:
        if not self.alive:
            return False
        empty_fields = np.argwhere(self.grid == 0)
        if len(empty_fields) == 0:
            self._alive = False
            return False
        ij = empty_fields[self.rnd.integers(low=0, high=len(empty_fields))]
        x = self.rnd.choice([1,2], p=[0.8, 0.2])
        self.grid[*ij] = x
        self.history[-1][1, *ij] = 1
        if len(self.get_moves()) == 0:
            self._alive = False
            return False
        return True

    def try_move(self, action: Action) -> bool:
        if not self.alive:
            return False
        moves = 0
        grid_history = np.zeros(shape=(2,*self.grid.shape), dtype=self.grid.dtype)
        grid_history[0,:,:] = self.grid
        self.score_history.append(int(self.score))
        if self.save_random_state:
            self._rnd_history.append(self.rnd.bit_generator.state)
        for i1, y1 in enumerate(action.yrange(self.grid.shape[0])):
            for j1, x1 in enumerate(action.xrange(self.grid.shape[1])):
                if self.grid[y1,x1] == 0:
                    continue
                if action.axis == "y":
                    y2 = None
                    for yy in action.yrange(self.grid.shape[0])[:i1][::-1]: # Go as far up or down as possible until hitting another block
                        if self.grid[yy,x1] == 0:
                            y2 = yy
                            continue
                        elif self.grid[yy,x1] == self.grid[y1,x1] and grid_history[1,yy,x1] < self.grid.size + 2: # Merge
                            moves += 1
                            grid_history[1,yy,x1] = self.xyt_to_idx(y1, x1, 1)
                            self.grid[yy,x1] = self.grid[yy,x1]+1
                            self.grid[y1,x1] = 0
                            self.score += int(2**(self.grid[yy,x1]))
                            break
                        else:
                            break
                    if y2 is not None:
                        moves += 1
                        grid_history[1,y2,x1] = self.xyt_to_idx(y1, x1, 0)
                        self.grid[y2,x1] = self.grid[y1,x1]
                        self.grid[y1,x1] = 0        

                else:
                    x2 = None
                    for xx in action.xrange(self.grid.shape[1])[:j1][::-1]: # Go as far up or down as possible until hitting another block
                        if self.grid[y1,xx] == 0:
                            x2 = xx
                            continue
                        elif self.grid[y1,xx] == self.grid[y1,x1] and grid_history[1,y1,xx] < self.grid.size + 2: # Merge
                            moves += 1
                            grid_history[1,y1,xx] = self.xyt_to_idx(y1, x1, 1)
                            self.grid[y1,xx] = self.grid[y1,xx]+1
                            self.grid[y1,x1] = 0
                            self.score += int(2**(self.grid[y1,xx]))
                            break
                        else:
                            break
                    if x2 is not None:
                        moves += 1
                        grid_history[1,y1,x2] = self.xyt_to_idx(y1, x1, 0)
                        self.grid[y1,x2] = self.grid[y1,x1]
                        self.grid[y1,x1] = 0
        if moves == 0:
            return False
        self.history.append(grid_history)
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
    
    def undo(self) -> bool:
        if len(self.history) <= 1:
            return False
        if not self.alive:
            self._alive = True
        
        self.score = self.score_history.pop()
        self.grid = self.history.pop()[0]
        if len(self._rnd_history) > 1:
            self.rnd.bit_generator.state = self._rnd_history.pop()
        return True

    
    def plot_on_axis(self, ax: Axes, clear: bool = True, plot_arrows: bool = False):
        if clear:
            ax.clear()
            [p.remove() for p in reversed(ax.patches)]
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

                if plot_arrows and self.history[-1][1,y,x] == 1:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), width=1, height=1, color="red", fill=False))
                elif plot_arrows and self.history[-1][1,y,x] >= 2:
                    y0, x0, t = self.idx_to_xyt(self.history[-1][1,y,x])

                    ax.add_patch(Arrow(x0, y0, (x-x0), (y-y0), color="red" if t == 1 else "blue", width=0.5, alpha=0.3))

    def __repr__(self) -> str:
        return f"<2048 Game{' (Ended)' if not self.alive else ''}: score {self.score}; {self.move_count} moves lead to {self.highest_tile} as highest tile>"
