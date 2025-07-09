from matplotlib.pylab import Generator
from ..game import *

import gymnasium as gym
from typing import Any, Sequence, SupportsFloat, Optional, Self

class Actions2048(gym.spaces.Space):

    def __init__(self):
        self.actions = np.array([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])
        super().__init__()

    def sample(self, mask: Any | None = None, probability: Any | None = None) -> Any:
        if mask is not None and probability is not None:
            raise ValueError(f"Only one of `mask` or `probability` can be provided, actual values: mask={mask}, probability={probability}")
        if mask is not None:
            return np.random.choice(self.actions[mask])
        return np.random.choice(self.actions, p=probability)


class Env2048(gym.Env):
    def __init__(self, shape: tuple[int, int] = (4, 4)) -> None:
        super().__init__()
        self._game = Game(shape=shape)
        self.action_space = Actions2048()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=16,
            shape=shape,
            dtype=np.uint8,
        )

    @property
    def game(self) -> Game:
        return self._game

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
            Returns:
                tuple: (observation, reward, terminated, truncated, info)
        """
        self.game.try_move(action)
        return self._get_obs(), self.game.reward, not self.game.alive, False, self._get_info()
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
            Returns:
                tuple: (observation, info)
        """
        super().reset(seed=seed)
        self._game = Game(generator_or_seed=self.np_random)
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return self.game.grid.copy()
    
    def _get_info(self) -> dict[str, Any]:
        return {
            "alive": self.game.alive,
            "score": self.game.score,
            "highest_tile": self.game.highest_tile,
            "move_count": self.game.move_count,
        }
    

