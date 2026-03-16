import functools
import os
import os.path as osp
import random
import re
import sys
import time
from collections import deque
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import debug
from mujoco_playground import registry

from gpe.utils import acting
from gpe.utils.buffer import (
    RunningStatistics,
    RunningStatisticsState,
    UniformSamplingQueue,
)
from gpe.utils.logger import EpochLogger
from gpe.utils.models import EnsembleValue, MHPolicy, get_tree_norm
from gpe.utils.types import Transition
from gpe.utils.utils import make_static_config_from_dict, single_agent_args

# jax.config.update("jax_disable_jit", True)
EPS = 1e-6

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "update_tau": 0.005,
    "weight_decay": 0.01,
    "episode_length": 1000,
    "warmup_samples": int(1e4),
    "max_replay_size": int(2e5),
    "total_iteration": int(1e6),
}


def polyak_update(target_model, curr_model, tau):
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)

    new_target_param = jax.tree_util.tree_map(
        lambda t, c: (1 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target_param)
    return target_model


@jax.jit
def discounted_sum(arr, gamma):
    dtype = arr.dtype
    horizon = arr.shape[0]

    def body_fun(t, cumsum):
        cumsum = arr[horizon - 1 - t] + gamma * cumsum
        return cumsum

    init_cumsum = jnp.zeros_like(arr[0], dtype=dtype)
    cumsum = jax.lax.fori_loop(0, horizon, body_fun, init_cumsum)
    return cumsum


def prefill_buffer(
    key,
    env,
    env_state,
    buffer_state,
    policy,
    buffer,
    num_itr,
):
    def body(carry, unused):
        key, env_state, buffer_state = carry
        key, subkey = jax.random.split(key)
        env_state, buffer_state = get_experience(
            key=subkey,
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            policy=policy,
            buffer=buffer,
        )
        return (key, env_state, buffer_state), ()

    jitted_body = jax.jit(body)
    (_, env_state, buffer_state), () = jax.lax.scan(
        jitted_body,
        (key, env_state, buffer_state),
        (),
        length=num_itr,
    )
    return env_state, buffer_state


def get_experience(
    key,
    env,
    env_state,
    buffer_state,
    policy,
    buffer,
):
    env_state, n_transition = acting.actor_step(
        env=env,
        env_state=env_state,
        policy=policy,
        key=key,
        extra_fields=("truncation",),
    )
    buffer_state = buffer.insert(buffer_state, n_transition)
    return env_state, buffer_state


def value_loss_grad_fun(
    target_value_model,
    value_model,
    policy_model,
    data,
    gamma,
    key,
):
    batch, num_env, obs_dim = data.observation.shape
    batch, num_env, act_dim = data.action.shape

    # Batch X obs/act/()_dim
    obs = data.observation.reshape(batch * num_env, obs_dim)
    act = data.action.reshape(batch * num_env, act_dim)
    next_obs = data.next_observation.reshape(batch * num_env, obs_dim)
    reward = data.reward.reshape(batch * num_env)
    done = data.discount.reshape(batch * num_env)

    # Batch X Horizon
    next_act, _ = policy_model(next_obs, act, key)
    # Batch
    next_q = jnp.minimum(*target_value_model(jnp.concat([next_obs, next_act], axis=-1)))
    target_v = reward + gamma * done * next_q

    def loss_fun(value_model):
        # Batch X Horizon X obs_act_dim
        target_oa = jnp.concat([obs, act], axis=-1)
        # Batch X Horizon
        pred_v1, pred_v2 = value_model(target_oa)
        v1_loss = optax.huber_loss(pred_v1, target_v, delta=2.0)
        v2_loss = optax.huber_loss(pred_v2, target_v, delta=2.0)
        loss = jnp.mean(v1_loss) + jnp.mean(v2_loss)
        return loss

    grad_fun = nnx.value_and_grad(loss_fun)
    loss, grads = grad_fun(value_model)

    return loss, grads


def policy_loss_grad_fun(
    value_model,
    policy_model,
    data,
    config,
    key,
):
    batch, num_env, obs_dim = data.observation.shape
    batch, num_env, act_dim = data.action.shape

    # Batch X obs/act/()_dim
    obs = data.observation.reshape(batch * num_env, obs_dim)
    act = data.action.reshape(batch * num_env, act_dim)
    pi_act, _ = policy_model(obs, act, key)

    q = jnp.minimum(*value_model(jnp.concat([obs, act], axis=-1)))
    v = jnp.minimum(*value_model(jnp.concat([obs, pi_act], axis=-1)))
    adv = q - v

    def loss_fun(policy_model):
        h = policy_model.h(obs, act)
        hpi = policy_model.h(obs, pi_act)
        pg_loss = jnp.mean(adv * (h - hpi))
        reg_loss = config.lmbda * jnp.mean(h**2 + hpi**2)
        return pg_loss + reg_loss, (pg_loss, reg_loss)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(policy_model)

    return loss, grads, *aux_values, q.mean(), v.mean()


def train_step(
    value_model_target,
    value_model,
    value_optimizer,
    policy_model,
    policy_optimizer,
    batch_data,
    config,
    key,
    steps,
):
    value_key, policy_key = jax.random.split(key)

    value_loss, value_grads = value_loss_grad_fun(
        target_value_model=value_model_target,
        value_model=value_model,
        policy_model=policy_model,
        data=batch_data,
        gamma=config.gamma,
        key=value_key,
    )
    value_optimizer.update(value_grads)

    policy_loss, policy_grads, *policy_aux = policy_loss_grad_fun(
        value_model=value_model,
        policy_model=policy_model,
        data=batch_data,
        config=config,
        key=policy_key,
    )
    policy_optimizer.update(policy_grads)

    value_model_target = polyak_update(
        value_model_target, value_model, config.update_tau
    )

    return (
        value_loss,
        policy_loss,
        *policy_aux,
    )


@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    value_model_target,
    value_model,
    value_optimizer,
    policy_model,
    policy_optimizer,
    config,
    key,
):
    num_steps = config.log_freq

    def body_fun(i, carry):
        (
            _,
            env_state,
            buffer_state,
            running_state,
            value_model_target,
            value_model,
            value_optimizer,
            policy_model,
            policy_optimizer,
            key,
        ) = carry

        key, buffer_key, train_key = jax.random.split(key, 3)

        env_state, buffer_state = get_experience(
            key=buffer_key,
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            policy=policy_model,
            buffer=buffer,
        )

        running_state = RunningStatistics.insert_reward(running_state, env_state.reward)

        buffer_state, batch_data = buffer.sample(buffer_state)

        val = train_step(
            value_model_target=value_model_target,
            value_model=value_model,
            value_optimizer=value_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            batch_data=batch_data,
            config=config,
            key=train_key,
            steps=i,
        )

        return (
            val,
            env_state,
            buffer_state,
            running_state,
            value_model_target,
            value_model,
            value_optimizer,
            policy_model,
            policy_optimizer,
            key,
        )

    init_val = (jnp.zeros((), dtype=jnp.float32),) * 6
    init_carry = (
        init_val,
        env_state,
        buffer_state,
        running_state,
        value_model_target,
        value_model,
        value_optimizer,
        policy_model,
        policy_optimizer,
        key,
    )
    val, env_state, buffer_state, running_state, *_ = nnx.fori_loop(
        0, num_steps, body_fun, init_carry
    )

    return *val, env_state, buffer_state, running_state, num_steps


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)

    prng_key = jax.random.PRNGKey(args.seed)
    rngs = nnx.Rngs(
        default=args.seed,
        params=args.seed + 3,
        dropout=args.seed + 5,
        random_sample=args.seed + 7,
    )

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    config = default_cfg
    config["beta"] = args.beta
    config["lmbda"] = args.lmbda
    config["decay"] = args.decay
    config["num_integral_steps"] = args.num_integral_steps
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation
    config["lr"] = args.lr

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")
    config["batch_size"] = batch_size

    config_data = make_static_config_from_dict(name="State", d=config)()

    # training environment
    prng_key, env_key, eval_env_key = jax.random.split(prng_key, 3)
    env_key = jax.random.split(env_key, 1)
    env = acting.wrap_env_for_training(
        env=registry.load(args.task),
        episode_length=config["episode_length"],
    )
    env_state = env.reset(env_key)

    # set model
    obs_dim, act_dim = env.observation_size, env.action_size
    policy_model = MHPolicy(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        beta=config["beta"],
        hidden_size=config["hidden_size"],
        decay=config["decay"],
        num_itr=config["num_integral_steps"],
    )
    policy_optimizer = nnx.Optimizer(
        model=policy_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )

    value_model = EnsembleValue(
        rngs=rngs,
        x_dim=obs_dim + act_dim,
        hidden_size=config["hidden_size"],
    )
    value_optimizer = nnx.Optimizer(
        model=value_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )
    value_model_target = deepcopy(value_model)

    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1, act_dim))
    dummy_zero = jnp.zeros((1,))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=dummy_zero,
        discount=dummy_zero,
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": dummy_zero}},
    )

    buffer = UniformSamplingQueue(
        max_replay_size=config["max_replay_size"],
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size,
    )
    prng_key, buffer_key = jax.random.split(prng_key)
    buffer_state = buffer.init(buffer_key)

    prng_key, running_key = jax.random.split(prng_key)
    running_state = RunningStatistics.init((config["episode_length"],), running_key)

    # set logger
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)
    logger.log("Start prefilling buffer")

    prng_key, buffer_key = jax.random.split(prng_key)
    env_state, buffer_state = prefill_buffer(
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        policy=policy_model,
        buffer=buffer,
        num_itr=config["warmup_samples"],
    )

    logger.log("Start training value and policy model")
    steps = buffer.size(buffer_state)
    while steps < config["total_iteration"]:
        prng_key, subkey = jax.random.split(prng_key)

        val = train_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            running_state=running_state,
            value_model_target=value_model_target,
            value_model=value_model,
            value_optimizer=value_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            config=config_data,
            key=subkey,
        )

        (
            value_loss,
            policy_loss,
            policy_pg_loss,
            policy_reg_loss,
            policy_qmean,
            policy_vmean,
            env_state,
            buffer_state,
            running_state,
            num_steps,
        ) = val

        steps += num_steps

        logger.logged = False

        if (steps % config["log_freq"] == 0) and (not logger.logged):
            logger.log_tabular("Train/Steps", steps)

            logger.log_tabular("Loss/Loss_value", value_loss.item())

            logger.log_tabular("Loss/Loss_policy", policy_loss.item())
            logger.log_tabular("Loss/Loss_policy_pg", policy_pg_loss.item())
            logger.log_tabular("Loss/Loss_policy_reg", policy_reg_loss.item())

            logger.log_tabular("Loss/policy_q_value", policy_qmean.item())
            logger.log_tabular("Loss/policy_v_value", policy_vmean.item())

            logger.log_tabular(
                "Norm/value_model",
                get_tree_norm(nnx.state(value_model, nnx.Param)),
            )
            logger.log_tabular(
                "Norm/policy_model",
                get_tree_norm(nnx.state(policy_model, nnx.Param)),
            )

            logger.log_tabular("Eval/Return", running_state.reward_state.data.sum())

            logger.dump_tabular()

        if steps % config["save_freq"] == 0:
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=value_model,
                prefix="value",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=policy_model,
                prefix="policy",
            )

        if steps >= config["total_iteration"]:
            break

    logger.nn_model_save(itr=steps, nn_model_saver_element=value_model, prefix="value")
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=policy_model, prefix="policy"
    )
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
