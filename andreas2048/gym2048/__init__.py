from .wrapper import Env2048

from gymnasium.envs.registration import register

register(
    id="andreas_2048",
    entry_point=Env2048, # type: ignore
    max_episode_steps=100000
)