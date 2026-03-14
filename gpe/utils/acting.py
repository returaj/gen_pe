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
import numpy as np

from gpe.utils import types


def actor_step(
    env: types.Env,
    env_state: types.EnvState,
    policy: types.Policy,
    key: types.PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[types.EnvState, types.Transition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.obs, key) # add initial action info
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, types.Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
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
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = actor_step(
            env, state, policy, current_key, extra_fields=extra_fields
        )
        return (nstate, next_key), transition

    f_jit = jax.jit(f, donate_argnums=(0,))
    (final_state, _), data = jax.lax.scan(
        f_jit, (env_state, key), (), length=unroll_length
    )
    return final_state, data
