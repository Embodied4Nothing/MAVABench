import os
import json
from MAVABench.utils.register import register
from MAVABench.envs.dm_env import LM4ManipDMEnv
from MAVABench.configs import name2config
from MAVABench.utils.utils import find_key_by_value

# load global robot config here, corresponding to different embodiments
with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/robot_config.json"), "r") as f:
    ROBOT_CONFIG= json.load(f)

with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_config.json"), "r") as f:
    TASK_CONFIG = json.load(f)

def load_env(task, 
             robots=["franka_1", "franka_2"],
             config=None, 
             time_limit=float('inf'), 
             reset_wait_step=10, 
             episode_config=None, 
             random_init=False,
             **kwargs):
    """
    load environment with given config
    params:
        task: str, name of the environment/task
        robots: list, names of the robots
        config: dict, additional configuration for the environment, including robot, task, etc.
        time_limit: int, maximum time steps for the environment
        reset_wait_step: int, number of steps to wait after reset, using for initialize the scene with no collision
        episode_config: dict, deterministic config for a specific episode, used for evaluation or trajetcory replay.
        random_init: bool, if true, the env will take random layout/texture in each reset. Set this value 'False' when eval or replay.
    """
    # load config
    task_series = find_key_by_value(name2config, task)
    specific_config = TASK_CONFIG.get(task_series, {})
    default_config = TASK_CONFIG["default"]
    default_config.update(specific_config)
    if config is not None and isinstance(config, dict):
        default_config.update(config)
    # load and update robot config first and then load robot entity
    robot_configs = []
    for robot in robots:
        robot_config = ROBOT_CONFIG.get(robot, None)
        assert robot_config is not None, f"robot {robot} is not supported"
        robot_config_overide = default_config.get("robot", {})
        robot_config.update(robot_config_overide)
        robot_configs.append(robot_config)

    robots = {name: register.load_robot(name)(**config) for name, config in zip(robots, robot_configs)}
    if default_config['task'] and default_config['task'].get("random_init", None) is not None:
        random_init = default_config['task']['random_init']
    if episode_config is not None:
        # forbid random initialization if given episode config
        random_init = False 
    task = register.load_task(task)(task, robots, episode_config=episode_config, random_init=random_init, **kwargs)
    env = LM4ManipDMEnv(task=task, time_limit=time_limit, reset_wait_step=reset_wait_step)
    env.reset()
    return env