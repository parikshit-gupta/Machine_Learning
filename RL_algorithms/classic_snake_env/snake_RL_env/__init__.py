from gymnasium.envs.registration import register

register(
    id="snake_RL_env/GridWorld-v0",
    entry_point="snake_RL_env.envs:GridWorldEnv",
)

register(
    id="snake_RL_env/ClassicSnake-v0",
    entry_point="snake_RL_env.envs:ClassicSnakeEnv",
)


