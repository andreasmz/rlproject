import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.image import AxesImage
from matplotlib.patches import Arrow, Rectangle
from typing import Literal, Callable
import time


class Action(Enum):
    UP = (0, "y", (-1, 0), 3)
    DOWN = (1, "y", (1, 0), 1)
    LEFT = (2, "x", (0,-1), 2)
    RIGHT = (3, "x", (0,1), 0)
    
    axis: Literal["x", "y"]
    dy: int
    dx: int
    rotations: int

    def __new__(cls, value, axis, dxy, rotations):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.axis = axis
        obj.dy = dxy[0]
        obj.dx = dxy[1]
        obj.rotations = rotations
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
    table_cache: dict[tuple[int,int], np.ndarray] = {}
    origin_table_cache: dict[tuple[int,int], np.ndarray] = {}
    score_table_cache: dict[tuple[int,int], np.ndarray] = {}

    def __init__(self, shape:tuple[int, int] = (4,4), generator_or_seed: np.random.Generator|int|None = None, persistent_rnd: bool = False) -> None:
        self.grid = np.zeros(shape=shape, dtype=np.uint8)
        if generator_or_seed is None or isinstance(generator_or_seed, int):
            self.rnd = np.random.default_rng(seed=generator_or_seed)
        else:
            self.rnd = generator_or_seed

        if shape in Game.table_cache:
            self.table = Game.table_cache[shape]
            self.origin_table = Game.origin_table_cache[shape]
            self.score_table = Game.score_table_cache[shape]
        else:
            self.build_table()

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

        self.try_spawn()
        self.try_spawn()

    def build_table(self) -> None:
        shape = self.grid.shape
        self.table = np.zeros(shape=[shape[0]*shape[1]+2 for x in range(shape[1])] + [shape[1]], dtype=np.uint8)
        self.origin_table = np.zeros(shape=[shape[0]*shape[1]+2 for x in range(shape[1])] + [shape[1]], dtype=np.int8)
        self.score_table = np.zeros(shape=[shape[0]*shape[1]+2 for x in range(shape[1])], dtype=np.int32)

        for row_tuple in np.indices(self.table.shape[:-1]).reshape(len(self.table.shape)-1, -1).T:
            score = 0
            self.table[*row_tuple] = row_tuple
            for i1 in range(len(row_tuple)-2,-1,-1):
                x1 = self.table[*row_tuple,i1]
                if x1 == 0:
                    continue
                i2_max = None
                for i2 in range(i1+1,len(row_tuple)):
                    x2 = self.table[*row_tuple,i2]
                    if x1 == x2 and self.origin_table[*row_tuple, i2] >= 0:
                        i2_max = None
                        score += 2**(int(x1)+1)
                        self.table[*row_tuple, i1] = 0
                        self.table[*row_tuple, i2] = x2+1
                        self.origin_table[*row_tuple, i2] = -i1-1
                        break
                    elif x2 == 0:
                        i2_max = i2
                        continue
                    break
                if i2_max is not None:
                    self.table[*row_tuple, i1] = 0
                    self.table[*row_tuple, i2_max] = x1
                    self.origin_table[*row_tuple, i2_max] = i1+1
            if (self.table[*row_tuple] == row_tuple).all():
                self.score_table[*row_tuple] = -1
            else:
                self.score_table[*row_tuple] = score
        Game.table_cache[shape] = self.table
        Game.origin_table_cache[shape] = self.origin_table
        Game.score_table_cache[shape] = self.score_table

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
    
    @property
    def grid_decoded(self) -> np.ndarray:
        r = 2**self.grid.astype(np.uint32)
        r[r == 1] = 0
        return r
        
    def try_spawn(self) -> bool:
        if not self.alive:
            return False
        empty_fields = np.argwhere(self.grid == 0)
        if len(empty_fields) == 0:
            self._alive = False
            return False
        ij = empty_fields[self.rnd.integers(low=0, high=len(empty_fields))]
        x = self.rnd.choice([1,2], p=[0.9, 0.1])
        self.grid[*ij] = x
        self.history[-1][1, *ij] = 63
        if len(self.get_moves()) == 0:
            self._alive = False
            return False
        return True
    
    def try_move(self, action: Action, no_spawn: bool = False) -> bool:
        if not self.alive:
            return False

        # History
        grid_history = np.zeros(shape=(2,*self.grid.shape), dtype=self.grid.dtype)
        grid_history[0,:,:] = self.grid
        self.score_history.append(int(self.score))
        if self.persistent_rnd:
            self._rnd_history.append(self.rnd.bit_generator.state)

        grid_r = np.rot90(self.grid, k=action.rotations)
        history_r = np.rot90(grid_history, k=action.rotations, axes=(1,2))

        # Find if there is at least one valid move
        for r in range(grid_r.shape[0]):
            if self.score_table[*grid_r[r]] >= 0:
                break
        else:
            return False
        
        for r in range(grid_r.shape[0]):
            history_r[1,r] = np.sign(self.origin_table[*grid_r[r]])*r + self.origin_table[*grid_r[r]]
            self.score += max(0, int(self.score_table[*grid_r[r]]))
            grid_r[r] = self.table[*grid_r[r]]
        
        #grid_history[1] = np.sign(grid_history[1]) * np.tile(np.arange(0, self.grid.size, self.grid.shape[1]).reshape(-1, 1), (1, self.grid.shape[1])) + grid_history[1]
        self.history.append(grid_history)
        if not no_spawn:
            self.try_spawn()
        
        return True
    
    def get_moves(self) -> list[Action]:
        possible_actions = []
        for a in Action:
            grid_r = np.rot90(self.grid, k=a.rotations)
            for r in range(grid_r.shape[0]):
                if self.score_table[*grid_r[r]] >= 0:
                    possible_actions.append(a)
                    break
        return possible_actions
    
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

    
    def plot_on_axis(self, ax: Axes, n: int = -1, clear: bool = True, plot_arrows: bool = False) -> AxesImage:
        if clear:
            ax.clear()
            [p.remove() for p in reversed(ax.patches)]
        grid = self.history[n]
        score = self.score_history[n]
        tile_history = self.tile_history[n]
        img = ax.imshow(self.history[n], cmap=Game.mpl_cmap, norm=Game.mpl_norm)
        ax.set_axis_off()
        if n != -1:
            ax.set_title(f"score {score} (step {n+1 if n >= 0 else len(self.history) + n + 1}/{len(self.history)})")
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
        return img
    
    def render_game(self, plot_arrows: bool = False, interval: int = 200) -> FuncAnimation:
        fig, ax = plt.subplots()
        def _draw(n):
            img = self.plot_on_axis(ax, n=n, clear=True, plot_arrows=(plot_arrows if n != -1 else False))
            return img,
        return FuncAnimation(fig=fig, func=_draw, frames=([-1] + [n for n in range(len(self.history))]), interval=interval, blit=True, repeat=False)

    def __repr__(self) -> str:
        return self.__call__(n=-1)

    def __call__(self, n: int = -1) -> str:
        if n == -1:
            return f"<2048 Game{' (Ended)' if not self.alive else ''}: score {self.score} and {self.move_count} moves lead to {self.highest_tile()} as highest tile>\n{str(self.grid_decoded())}"
        else:
            return f"<2048 Game{f' (step {n+1 if n >= 0 else len(self.history) + n + 1}/{len(self.history)})'}: score {self.score_history[n]} and {self.highest_tile(n)} as highest tile>\n{str(self.grid_decoded(n))}"