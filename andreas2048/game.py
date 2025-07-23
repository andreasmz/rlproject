import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.image import AxesImage
from matplotlib.patches import Arrow, Rectangle
from typing import Literal, Any, Self, Callable
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
    
class State:

    table_cache: dict[tuple[int,int], np.ndarray] = {}
    """ Caches the tables for different shapes """
    origin_table_cache: dict[tuple[int,int], np.ndarray] = {}
    """ Caches the tile origin for different shapes """
    score_table_cache: dict[tuple[int,int], np.ndarray] = {}
    """ Caches the score tables for different shapes """

    TILE_SPAWNED_CONST = 127

    def __init__(self, 
                 n: int, 
                 score: int, 
                 reward: float,
                 grid: np.ndarray, 
                 rnd: np.random.Generator,
                 alive: bool = True, 
                 tile_history: np.ndarray|None = None, 
                 action: Action|None = None,
                 parent_state: Self|None = None,
                 probs: tuple[list[int], list[float]] = ([1,2], [0.9, 0.1])
                 ) -> None:
        self.n = n
        self.grid = grid
        self.tile_history = tile_history if tile_history is not None else np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        self.score = score
        self.reward = reward
        self.alive = alive
        self.action = action
        self.rnd = rnd
        self.parent = parent_state
        self.probs = probs
        self.warm_start_score: int = 0

        shape = self.grid.shape
        if shape not in State.table_cache:
            State.build_table(shape)
        self._table = State.table_cache[shape]
        self._origin_table = State.origin_table_cache[shape]
        self._score_table = State.score_table_cache[shape]

    def warm_start(self, tiles: list[int], p: list[float], n:int = 2) -> "State":
        if len(tiles) != len(p):
            raise ValueError(f"The list of tiles and p must have the same number of elements. You provied {len(tiles)} tiles and {len(p)} p")
        self.grid = np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        self.tile_history = np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        for i in range(n):
            self.apply_spawn(probs=(tiles, np.array(p)/np.sum(p)))
        self.score = self.estimate_score()
        self.warm_start_score = self.score
        return self

    def estimate_score(self) -> int:
        x = self.grid.astype(np.int32)
        return int(round(np.sum(np.clip(2**x*(x-1.1), a_min=0, a_max=None))))

    @property
    def highest_tile(self) -> int:
        """ Returns the highest tile (as decoded value, e.g. 2048) in the state """
        return 2**int(np.max(self.grid))
    
    @property
    def grid_stacks(self) -> np.ndarray:
        """ Converts the current grid from (h,w) to (i,h,w) for each of the possible tile values"""
        return (self.grid[None, :, :] == np.arange(1, self.grid.size+1)[:, None, None]).astype(self.grid.dtype)
    
    @property
    def flat_stack(self) -> np.ndarray:
        return self.grid_stacks.flatten()
    
    @property
    def grid_decoded(self) -> np.ndarray:
        """ Decodes the log2 into readable values (e.g. 12 -> 2048)"""
        r = 2**self.grid.astype(np.uint32)
        r[r == 1] = 0
        return r
    
    def get_moves(self) -> list[Action]:
        possible_actions = []
        for a in Action:
            grid_r = np.rot90(self.grid, k=a.rotations)
            for r in range(grid_r.shape[0]):
                if self._score_table[*grid_r[r]] >= 0:
                    possible_actions.append(a)
                    break
        return possible_actions
    
    def apply_move(self, action: Action) -> "State":
        """
        Returns the next state given an action. If the move is invalid, return this state
        """
        if not self.alive:
            return self
        
        grid = self.grid.copy()
        tile_history = np.zeros(shape=self.grid.shape, dtype=np.int8)
        score = self.score

        grid_r = np.rot90(grid, k=action.rotations)
        tile_history_r = np.rot90(tile_history, k=action.rotations)
        index_r = np.rot90(np.arange(1,grid.size+1).reshape(grid.shape), k=action.rotations).reshape(-1) # Track how indices are rotated

        # Find if there is at least one valid move
        for r in range(grid_r.shape[0]):
            if self._score_table[*grid_r[r]] >= 0:
                break
        else:
            return self
        
        # Process row transformations
        for r in range(grid_r.shape[0]):
            for i, th in enumerate(self._origin_table[*grid_r[r]]):
                if th != 0 and th != State.TILE_SPAWNED_CONST:
                    th = np.sign(th)*index_r[np.abs(th) + r*grid_r.shape[1] - 1] # Transform the origin index
                tile_history_r[r, i] = th
            score += max(0, int(self._score_table[*grid_r[r]])) # While some row moves do not change the grid, the overall Action may nevertheless be valid --> max(0,score)
            grid_r[r] = self._table[*grid_r[r]] # Update grid
        
        return State(n=(self.n+1), score=score, reward=score-self.score, grid=grid, tile_history=tile_history, action=action, rnd=self.rnd, parent_state=self, probs=self.probs)
    
    def apply_spawn(self, probs:tuple[list[int], list[float]]|None = None) -> "State":
        """ Tries to spawn a new tile in this state. Set alive to False and returns False if no tile can be spawned or the move count is zero after spawning"""
        if not self.alive:
            return self
        empty_fields = np.argwhere(self.grid == 0)
        if len(empty_fields) == 0:
            self.alive = False
            return self
        ij = empty_fields[self.rnd.integers(low=0, high=len(empty_fields))]
        x = self.rnd.choice(probs[0] if probs is not None else self.probs[0], p=(probs[1] if probs is not None else self.probs[1]))
        self.grid[*ij] = x
        self.tile_history[*ij] = State.TILE_SPAWNED_CONST
        if len(self.get_moves()) == 0:
            self.alive = False
        return self
    
    def next(self, action: Action) -> "State":
        """ Returns the next state. If the move is invalid, returns self"""
        s = self.apply_move(action=action)
        if s.n != self.n:
            s.apply_spawn()
        return s
    
    def get_next_states(self, action: Action) -> "dict[State, float]":
        """ Returns all next states with their probability """
        s = self.apply_move(action=action)
        if s == self:
            return {self: 1.0}
        empty_fields = np.argwhere(s.grid == 0)
        if len(empty_fields) == 0:
            return {self: 1.0}
        r: dict[State, float] = {}
        for ij in empty_fields:
            for v,p in zip(*self.probs):
                sn = s.clone()
                sn.grid[*ij] = v
                sn.tile_history[*ij] = State.TILE_SPAWNED_CONST
                if len(sn.get_moves()) == 0:
                    sn.alive = False
                r[sn] = p/len(empty_fields)
        return r
    
    def backtrace_reward(self, discounted_reward: float, lambda_: float):
        self.reward += discounted_reward
        if self.parent is not None:
            self.parent.backtrace_reward(discounted_reward=(discounted_reward*lambda_), lambda_=lambda_)

    @property
    def score_bonus(self) -> float:
        bonus_grid = np.zeros(shape=self.grid.shape, dtype=np.int32)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i,j] < 3:
                    continue
                if i < (self.grid.shape[0] - 1):
                    if self.grid[i,j] == (self.grid[i+1,j]):
                        bonus_grid[i,j] = 2**self.grid[i,j]
                    elif self.grid[i,j] == (self.grid[i+1,j] + 1) and self.grid[i+1,j] >= 3:
                        bonus_grid[i,j] = 2**self.grid[i,j]
                    elif (self.grid[i,j] + 1) == self.grid[i+1,j]:
                        bonus_grid[i+1,j] = 2**self.grid[i+1,j]
                if j < (self.grid.shape[1] - 1):
                    if self.grid[i,j] == (self.grid[i,j+1]):
                        bonus_grid[i,j] = 2**self.grid[i,j]
                    elif self.grid[i,j] == (self.grid[i,j+1] + 1) and self.grid[i,j+1] >= 3:
                        bonus_grid[i,j] = 2**self.grid[i,j]
                    elif (self.grid[i,j] + 1) == self.grid[i,j+1]:
                        bonus_grid[i,j+1] = 2**self.grid[i,j+1]
        return np.sum(bonus_grid)


    
    @staticmethod
    def build_table(shape: tuple[int, int]) -> None:
        """ Build the tables for a given shape """
        table = np.zeros(shape=[shape[0]*shape[1]+2 for x in range(shape[1])] + [shape[1]], dtype=np.uint8)
        origin_table = np.zeros(shape=[shape[0]*shape[1]+2 for x in range(shape[1])] + [shape[1]], dtype=np.int8)
        score_table = np.zeros(shape=[shape[0]*shape[1]+2 for x in range(shape[1])], dtype=np.int32)

        for row_tuple in np.indices(table.shape[:-1]).reshape(len(table.shape)-1, -1).T: # Itterate over every tuple
            score = 0
            table[*row_tuple] = row_tuple.astype(table.dtype) # Map first s -> s
            for i1 in range(len(row_tuple)-2,-1,-1): # Iterate from 2nd most right field to the left
                x1 = table[*row_tuple,i1]
                if x1 == 0:
                    continue
                i2_max = None
                for i2 in range(i1+1,len(row_tuple)): # Search for empty or equal tiles i2 on the right side of the current tile i1
                    x2 = table[*row_tuple,i2]
                    if x1 == x2 and origin_table[*row_tuple, i2] >= 0: # Merge if the tile has not been merged yet
                        i2_max = None
                        score += 2**(int(x1)+1)
                        table[*row_tuple, i1] = 0
                        table[*row_tuple, i2] = x2+1
                        origin_table[*row_tuple, i2] = -i1-1 # Indicate that a merge happened (Note: The index is shifted by 0 s.t. counting starts on -1)
                        break
                    elif x2 == 0: # If i2 is empty tile, remember highest i2 for move
                        i2_max = i2
                        continue
                    break
                if i2_max is not None:
                    table[*row_tuple, i1] = 0
                    table[*row_tuple, i2_max] = x1
                    origin_table[*row_tuple, i2_max] = i1+1 # Indicate that a merge happend (Note: The index is shifted by 0 s.t. counting starts on 1)
            if (table[*row_tuple] == row_tuple).all():
                score_table[*row_tuple] = -1 # Mark an invalid move with a negative score
            else:
                score_table[*row_tuple] = score
        State.table_cache[shape] = table
        State.origin_table_cache[shape] = origin_table
        State.score_table_cache[shape] = score_table

    # def __eq__(self, value: object) -> bool:
    #     if not isinstance(value, State):
    #         raise TypeError(f"Can't compare a state to {type(value)}")
    #     return self.n == value.n

    def clone(self, rot: int|None = None) -> "State":
        grid = self.grid.copy()
        if rot is not None:
            grid = np.rot90(grid, k=rot)
        return State(n=self.n, score=self.score, reward=self.reward, grid=grid, rnd=self.rnd, alive=self.alive, tile_history=None, action=self.action, parent_state=self.parent, probs=self.probs)
    
    def __repr__(self) -> str:
        return f"<2048 Game State n={self.n}: score: {self.score} - highest tile {self.highest_tile} >\n{str(self.grid_decoded)}"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        dels = ["rnd", "_table", "_origin_table", "_score_table"]
        for d in dels:
            if d in state:
                del state[d]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

class Game:
    """ Implements the 2048 game """

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

    def __init__(self, shape:tuple[int, int] = (4,4), generator_or_seed: np.random.Generator|int|None = None, persistent_rnd: bool = False) -> None:
        if generator_or_seed is None or isinstance(generator_or_seed, int):
            self.rnd = np.random.default_rng(seed=generator_or_seed)
        else:
            self.rnd = generator_or_seed
        self.persistent_rnd = persistent_rnd 

        self.shape = shape
        s0 = State(n=0, grid=np.zeros(shape=shape, dtype=np.uint8), tile_history=np.zeros(shape=shape, dtype=np.uint8), score=0, reward=0, rnd=self.rnd).apply_spawn().apply_spawn()
        self.history: list[State] = [s0]

    @property
    def state(self) -> State:
        return self.history[-1]
    
    @property
    def move_count(self) -> int:
        return len(self.history) - 1
    
    @property
    def grid(self) -> np.ndarray:
        return self.state.grid
    
    @property
    def alive(self) -> bool:
        return self.state.alive
    
    @property
    def score(self) -> int:
        return self.state.score
    
    @property
    def highest_tile(self) -> int:
        return self.state.highest_tile
    
    @property
    def grid_decoded(self) -> np.ndarray:
        return self.state.grid_decoded
    
    def get_moves(self) -> list[Action]:
        return self.state.get_moves()
    
    def next(self, action: Action) -> State:
        s = self.state.next(action=action)
        if s.n != self.state.n:
            self.history.append(s)
        return s
    
    def undo(self) -> bool:
        if len(self.history) <= 1:
            return False
        self.history.pop()
        return True
    
    def reward(self, n: int = -1) -> int|float:
        if n <= 0:
            n = len(self.history) + n
        return self.history[n].reward
    
    def plot_on_axis(self, ax: Axes, n: int = -1, clear: bool = True, plot_arrows: bool = False, value_str: str|None = None) -> AxesImage:
        if clear:
            ax.clear()
            [p.remove() for p in reversed(ax.patches)]
        grid = self.history[n].grid
        grid_decoded = self.history[n].grid_decoded
        score = self.history[n].score
        tile_history = self.history[n].tile_history
        img = ax.imshow(grid, cmap=Game.mpl_cmap, norm=Game.mpl_norm)
        ax.set_axis_off()
        s = ""
        if n != -1:
            s = (f"score {score} (step {n+1 if n >= 0 else len(self.history) + n + 1}/{len(self.history)})")
        elif self.history[n].alive:
            s = (f"score: {score}")
        else:
            s = (f"Game over (score: {score})")
        if value_str is not None:
            s += f"\n{value_str}"
        ax.set_title(s)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y,x] != 0:
                    c = "white" if grid[y,x] >= 3 else "black"
                    fsize = 26 if grid[y,x] <= 6 else 20
                    plt.text(x, y, grid_decoded[y,x], ha="center", va="center", color=c, fontsize=fsize)

                if plot_arrows:
                    th = tile_history[y,x]
                    if th == State.TILE_SPAWNED_CONST:
                        ax.add_patch(Rectangle((x-0.5, y-0.5), width=1, height=1, color="red", fill=False))
                    elif th < 0:
                        y0, x0 = (-th-1) // grid.shape[1], (-th-1) % grid.shape[1]
                        ax.add_patch(Arrow(x0, y0, (x-x0), (y-y0), color="red", width=0.5, alpha=0.3))
                    elif th > 0:
                        y0, x0 = (th-1) // grid.shape[1], (th-1) % grid.shape[1]
                        ax.add_patch(Arrow(x0, y0, (x-x0), (y-y0), color="blue", width=0.5, alpha=0.3))
        return img
    
    def render_game(self, plot_arrows: bool = False, interval: int = 200, value_func: Callable|None = None, state_func: Callable|None = None) -> FuncAnimation:
        fig, ax = plt.subplots()
        def _draw(n):
            value_str = value_func(self.history[n].grid) if value_func is not None else None
            value_str = state_func(self.history[n]) if state_func is not None else None
            img = self.plot_on_axis(ax, n=n, clear=True, plot_arrows=(plot_arrows if n != -1 else False), value_str=value_str)
            return img,
        return FuncAnimation(fig=fig, func=_draw, frames=([-1] + [n for n in range(len(self.history))]), interval=interval, blit=True, repeat=False)

    def __repr__(self) -> str:

        return f"<2048 Game{' (Ended)' if not self.alive else ''}: score: {self.score} - moves: {self.move_count} - highest tile: {self.highest_tile}>\n{str(self.grid_decoded)}"

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        dels = ["rnd"]
        for d in dels:
            if d in state:
                del state[d]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)