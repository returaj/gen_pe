import argparse
from distutils.util import strtobool

from flax import struct


def get_params_norm(params, grads=False):
    total_norm = 0.0
    for p in params:
        if grads:
            total_norm += p.grad.detach().data.norm(2).item()
        else:
            total_norm += p.data.norm(2).item()
    return total_norm


def make_static_config_from_dict(name: str, d: dict):
    annotations = {}
    defaults = {}

    for k, v in d.items():
        annotations[k] = type(v)
        defaults[k] = struct.field(
            default=v,
            pytree_node=False,  # made it fixed / immutable ?
        )

    cls = type(
        name,
        (),
        {
            "__annotations__": annotations,
            **defaults,
        },
    )

    return struct.dataclass(cls)


def single_agent_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {
            "name": "--use-eval",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "Use evaluation environment for testing",
        },
        {
            "name": "--save-video",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "Use evaluation environment for testing",
        },
        {
            "name": "--task",
            "type": str,
            "default": "OfflinePointGoal1Gymnasium-v0",
            "help": "The task to run",
        },
        {
            "name": "--experiment",
            "type": str,
            "default": "equal",
            "help": "Experiment name",
        },
        {
            "name": "--log-dir",
            "type": str,
            "default": "dsrl_model/runs",
            "help": "directory to save agent logs",
        },
        {
            "name": "--device",
            "type": str,
            "default": "cpu",
            "help": "The device to run the model on",
        },
        {
            "name": "--device-id",
            "type": int,
            "default": 0,
            "help": "The device id to run the model on",
        },
        {
            "name": "--write-terminal",
            "type": lambda x: bool(strtobool(x)),
            "default": True,
            "help": "Toggles terminal logging",
        },
        {
            "name": "--batch-size",
            "type": int,
            "default": 128,
            "help": "The number of steps to run in each environment per policy rollout",
        },
        {
            "name": "--lr",
            "type": float,
            "default": 1e-5,  # 1e-3 performs better
            "help": "Default common learning rate for the models",
        },
        {
            "name": "--lmbda",
            "type": float,
            "default": None,  # 1e-3 performs better
            "help": "hyperparameter lambda value",
        },
        {
            "name": "--normalize-observation",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "To normalize the state observation.",
        },
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    cfg_env = {}
    return args, cfg_env
