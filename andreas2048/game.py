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
    action_space = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def __init__(self, shape:tuple[int, int] = (4,4), generator_or_seed: np.random.Generator|int|None = None, persistent_rnd: bool = False) -> None:
        self.grid = np.zeros(shape=shape, dtype=np.uint8)
        if generator_or_seed is None or isinstance(generator_or_seed, int):
            self.rnd = np.random.default_rng(seed=generator_or_seed)
        else:
            self.rnd = generator_or_seed
        self.persistent_rnd = persistent_rnd
        self.score = 0
        self._alive = True
        self.history: list[np.ndarray] = [np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)]
        """ Tracks the history of the grid """
        self.tile_history: list[np.ndarray] = [np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)]
        """ Tracks the history of each tile per step: 0: Not moved, 1: spawned, 2:(n*m+2): moved origin, (n*m+2):2*(n*m+2): merged origin """
        self.score_history: list[int] = [self.score]
        """ Tracks the history of the score """
        self.action_history: list[Action] = []
        self._rnd_history = [self.rnd.bit_generator.state]

        self.xyt_to_idx = lambda y, x, t: y*self.grid.shape[0] + x + t*self.grid.size + 2 # Encodes (x,y,t) as int. Offset 2, as 0 and 1 have special meaning
        self.idx_to_xyt = lambda idx: ( (int((idx-2) % self.grid.size) // self.grid.shape[0]), int((idx-2) % self.grid.shape[0]) , int((idx-2) // self.grid.size) )

        self.try_spawn()
        self.try_spawn()

    @property
    def move_count(self) -> int:
        return len(self.history) - 1
    
    @property
    def alive(self) -> bool:
        return self._alive
    
    def highest_tile(self, n: int = -1):
        return 2**int(np.max(self.history[n]))
    
    def reward(self, n: int = -1) -> int:
        if n <= 0:
            n = len(self.history) + n
        return self.score_history[n] - self.score_history[max(0,n-1)]
    
    @property
    def grid_stacks(self) -> np.ndarray:
        return (self.grid[None, :, :] == np.arange(1, self.grid.size+1)[:, None, None]).astype(self.grid.dtype)
    
    @property
    def flat_stack(self) -> np.ndarray:
        return self.grid_stacks.flatten()
    
    def grid_decoded(self, n: int = -1) -> np.ndarray:
        if n == -1:
            r = 2**self.grid.astype(np.uint32)
        else:
            r = 2**self.history[n].astype(np.uint32)
        r[r == 1] = 0
        return r
    
    @property
    def max_possible_value(self) -> int:
        return self.grid.shape[0]*self.grid.shape[1] + 1
        
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
        self.history[-1][*ij] = x
        self.tile_history[-1][*ij] = 1
        if len(self.get_moves()) == 0:
            self._alive = False
            return False
        return True

    def try_move(self, action: Action) -> bool:
        if not self.alive:
            return False
        moves = 0
        tile_history = np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        history_n = self.history[0].copy()
        if self.persistent_rnd:
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
                        elif self.grid[yy,x1] == self.grid[y1,x1] and tile_history[yy,x1] < self.grid.size + 2: # Only allow to merge with not already merged tiles
                            moves += 1
                            tile_history[yy,x1] = self.xyt_to_idx(y1, x1, 1)
                            self.grid[yy,x1] = self.grid[yy,x1]+1
                            self.grid[y1,x1] = 0
                            self.score += int(2**(self.grid[yy,x1]))
                            break
                        else:
                            break
                    if y2 is not None:
                        moves += 1
                        tile_history[y2,x1] = self.xyt_to_idx(y1, x1, 0)
                        self.grid[y2,x1] = self.grid[y1,x1]
                        self.grid[y1,x1] = 0        

                else:
                    x2 = None
                    for xx in action.xrange(self.grid.shape[1])[:j1][::-1]: # Go as far up or down as possible until hitting another block
                        if self.grid[y1,xx] == 0:
                            x2 = xx
                            continue
                        elif self.grid[y1,xx] == self.grid[y1,x1] and tile_history[y1,xx] < self.grid.size + 2: # Merge
                            moves += 1
                            tile_history[y1,xx] = self.xyt_to_idx(y1, x1, 1)
                            self.grid[y1,xx] = self.grid[y1,xx]+1
                            self.grid[y1,x1] = 0
                            self.score += int(2**(self.grid[y1,xx]))
                            break
                        else:
                            break
                    if x2 is not None:
                        moves += 1
                        tile_history[y1,x2] = self.xyt_to_idx(y1, x1, 0)
                        self.grid[y1,x2] = self.grid[y1,x1]
                        self.grid[y1,x1] = 0
        if moves == 0:
            return False
        self.history.append(self.grid.copy())
        self.tile_history.append(tile_history)
        self.score_history.append(self.score)
        self.action_history.append(action)
        return self.try_spawn()
    
    def get_moves(self) -> list[Action]:
        all_actions = Game.action_space.copy()
        actions = []
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y,x] == 0:
                    continue
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
        
        self.history.pop()
        self.tile_history.pop()
        self.score_history.pop()
        self.grid = self.history[-1].copy()
        self.score = self.score_history[-1]
        if len(self._rnd_history) > 1:
            self.rnd.bit_generator.state = self._rnd_history.pop()
        return True

    
    def plot_on_axis(self, ax: Axes, n: int = -1, clear: bool = True, plot_arrows: bool = False):
        if clear:
            ax.clear()
            [p.remove() for p in reversed(ax.patches)]
        grid = self.history[n]
        score = self.score_history[n]
        tile_history = self.tile_history[n]
        ax.imshow(self.history[n], cmap=Game.mpl_cmap, norm=Game.mpl_norm)
        ax.set_axis_off()
        if n != -1:
            ax.set_title(f"score {score} (step {n if n >= 0 else len(self.history) + n + 1}/{len(self.history)})")
        elif self.alive:
            ax.set_title(f"score: {score}")
        else:
            ax.set_title(f"Game over (score: {score})")

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if grid[y,x] != 0:
                    c = "white" if grid[y,x] >= 3 else "black"
                    fsize = 26 if grid[y,x] <= 6 else 20
                    plt.text(x, y, 2**grid[y,x], ha="center", va="center", color=c, fontsize=fsize)

                if plot_arrows and tile_history[y,x] == 1:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), width=1, height=1, color="red", fill=False))
                elif plot_arrows and tile_history[y,x] >= 2:
                    y0, x0, t = self.idx_to_xyt(tile_history[y,x])

                    ax.add_patch(Arrow(x0, y0, (x-x0), (y-y0), color="red" if t == 1 else "blue", width=0.5, alpha=0.3))

    def __repr__(self) -> str:
        return self.__call__(n=-1)

    def __call__(self, n: int = -1) -> str:
        if n == -1:
            return f"<2048 Game{' (Ended)' if not self.alive else ''}: score {self.score} and {self.move_count} moves lead to {self.highest_tile()} as highest tile>\n{str(self.grid_decoded())}"
        else:
            return f"<2048 Game{f' (step {n+1 if n >= 0 else len(self.history) + n + 1}/{len(self.history)})'}: score {self.score_history[n]} and {self.highest_tile(n)} as highest tile>\n{str(self.grid_decoded(n))}"