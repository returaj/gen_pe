import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from absl import logging

from gpe.utils import acting, types

ReplayBuffer = Any
ReplayBufferState = Any


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: types.Params
    q_optimizer_state: optax.OptState
    q_params: types.Params
    target_q_params: types.Params
    gradient_steps: types.UInt64
    env_steps: types.UInt64
    alpha_optimizer_state: optax.OptState
    alpha_params: types.Params


def get_experience(
    replay_buffer: ReplayBuffer,
    env: types.Env,
    policy: types.Policy,
    env_state: types.EnvState,
    buffer_state: ReplayBufferState,
    key: types.PRNGKey,
) -> Tuple[
    types.EnvState,
    ReplayBufferState,
]:
    env_state, transitions = acting.actor_step(
        env, env_state, policy, key, extra_fields=("truncation",)
    )

    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return env_state, buffer_state

