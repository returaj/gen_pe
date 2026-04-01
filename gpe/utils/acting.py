# Copyright 2026 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper

from gpe.utils import types


class WrapPrevAction(wrapper.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._name = "init_action"

    def reset(self, rng: jnp.ndarray):
        batch = rng.shape[0]
        state = self.env.reset(rng)
        state.info[self._name] = jnp.zeros(
            shape=(batch, self.action_size), dtype=jnp.float32
        )
        return state

    def step(self, state: types.EnvState, action: types.Action):
        reset_init_action = jnp.zeros_like(state.info[self._name])
        batch = reset_init_action.shape[0]
        done = jnp.reshape(state.done, (batch, 1))
        init_action = jnp.where(done, reset_init_action, action)

        nstate = self.env.step(state, action)
        nstate.info[self._name] = init_action
        return nstate


def wrap_env_for_training(env, episode_length, action_repeat=1, full_reset=False):
    # Please see wrapper.wrap_for_brax_training() method. 
    # Since we need previous action in our state.info we have to do it individually. 
    env = brax_training.VmapWrapper(env)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = WrapPrevAction(env)
    env = wrapper.BraxAutoResetWrapper(env, full_reset=full_reset)
    return env


def actor_step(
    env: types.Env,
    env_state: types.EnvState,
    policy: types.Policy,
    key: types.PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[types.EnvState, types.Transition]:
    """Collect data."""

    # we need to expand dim as policy need B X horizon X dim
    # B X 1 X obs/act_dim
    init_action = jnp.expand_dims(env_state.info["init_action"], axis=1)
    obs = jnp.expand_dims(env_state.obs, axis=1)
    action, _ = policy(obs, init_action, key)  # add initial action info
    # B X act_dim
    action = jnp.squeeze(action, axis=1)
    n_env_state = env.step(env_state, action)
    state_extras = {x: n_env_state.info[x] for x in extra_fields}
    return (
        n_env_state,
        types.Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=env_state.obs,
            action=action,
            reward=n_env_state.reward,
            discount=1 - n_env_state.done,
            next_observation=n_env_state.obs,
            extras={"state_extras": state_extras},
        ),
    )


def generate_unroll(
    env: types.Env,
    env_state: types.EnvState,
    policy: types.Policy,
    key: types.PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[types.EnvState, types.Transition]:
    """Collect trajectories of given unroll_length."""

    def f(carry, unused_t):
        env_state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        n_env_state, n_transition = actor_step(
            env=env,
            env_state=env_state,
            policy=policy,
            key=current_key,
            extra_fields=extra_fields,
        )
        return (n_env_state, next_key), n_transition

    f_jit = jax.jit(f, donate_argnums=(0,))
    (final_state, _), data = jax.lax.scan(
        f_jit, (env_state, key), (), length=unroll_length
    )
    return final_state, data
