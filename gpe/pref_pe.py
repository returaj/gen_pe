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

from gpe.utils import acting, types
from gpe.utils.buffer import RunningStatistics, TrajectorySamplingQueue
from gpe.utils.logger import EpochLogger
from gpe.utils.models import EnsembleValue, MHPolicy, get_tree_norm
from gpe.utils.types import Transition
from gpe.utils.utils import make_static_config_from_dict, single_agent_args

# jax.config.update("jax_disable_jit", True)
EPS = 1e-6

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 5,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "update_tau": 0.005,
    "weight_decay": 0.01,
    "train_per_step": 2,
    "policy_update_freq": 2,
    "train_horizon": 10,
    "episode_length": 1000,
    "warmup_samples": int(1e3),
    "max_replay_size": int(6e5),
    "total_iteration": int(5e5),
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

    def body_fun(cumsum, t):
        cumsum = arr[horizon - 1 - t] + gamma * cumsum
        return (cumsum, cumsum)

    init_cumsum = jnp.zeros_like(arr[0], dtype=dtype)
    cumsum, vec_cumsum = jax.lax.scan(body_fun, init_cumsum, jnp.arange(horizon))
    return cumsum, jnp.flip(vec_cumsum, axis=0)


def reshape_transition(data):
    batch, horizon, num_env, obs_dim = data.observation.shape
    batch, horizon, num_env, act_dim = data.action.shape

    assert num_env == 1, f"number of env should be set to 1 not {num_env}"

    # Batch X H X obs/act/()_dim
    obs = data.observation.reshape(batch, horizon, obs_dim)
    act = data.action.reshape(batch, horizon, act_dim)
    next_obs = data.next_observation.reshape(batch, horizon, obs_dim)
    reward = data.reward.reshape(batch, horizon)
    discount = data.discount.reshape(batch, horizon)

    return types.Transition(
        observation=obs,
        action=act,
        next_observation=next_obs,
        reward=reward,
        discount=discount,
        extras=data.extras,
    )


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


def compute_target_value(value, data, pi_act_seq, config):
    gamma, lmbda = config.gamma, config.lmbda

    # B X 1 X obs_dim
    curr_obs = jnp.expand_dims(data.observation[:, 0], axis=1)
    # B X H+1 X obs_dim
    obs_seq = jnp.concatenate([curr_obs, data.next_observation], axis=1)

    # B X H+1
    v = jnp.minimum(*value(jnp.concatenate([obs_seq, pi_act_seq], axis=-1)))
    # B X H
    td = data.reward + gamma * data.discount * v[:, 1:] - v[:, :-1]
    # H X B
    _, adv_transpose = discounted_sum(td.T, gamma * lmbda)
    # B X H
    target_q = adv_transpose.T + v[:, :-1]
    return target_q


def value_loss_grad_fun(
    target_value_model,
    value_model,
    data,
    pi_act_seq,
    config,
):
    # Batch X H
    target_v = compute_target_value(target_value_model, data, pi_act_seq, config)

    def loss_fun(value_model):
        # Batch X Horizon X obs_act_dim
        target_oa = jnp.concat([data.observation, data.action], axis=-1)
        # Batch X Horizon
        pred_v1, pred_v2 = value_model(target_oa)
        v1_loss = optax.huber_loss(pred_v1, target_v, delta=2.0)
        v2_loss = optax.huber_loss(pred_v2, target_v, delta=2.0)
        loss = jnp.mean(v1_loss) + jnp.mean(v2_loss)
        # Batch
        priority = 0.5 * (jnp.abs(pred_v1 - target_v) + jnp.abs(pred_v2 - target_v))
        priority_loss = jnp.clip(priority[:, 0], max=1e4)
        return loss, (priority_loss,)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_value), grads = grad_fun(value_model)

    return loss, grads, *aux_value


def policy_loss_grad_fun(
    value_model,
    policy_model,
    data,
    pi_act_seq,
    config,
):
    # B X H X obs/act_dim
    obs = data.observation
    act = data.action
    pi_act = pi_act_seq[:, :-1]

    q = jnp.minimum(*value_model(jnp.concat([obs, act], axis=-1)))
    v = jnp.minimum(*value_model(jnp.concat([obs, pi_act], axis=-1)))
    adv = q - v

    def loss_fun(policy_model):
        h = policy_model.h(obs, act)
        hpi = policy_model.h(obs, pi_act)
        pg_loss = -jnp.mean(adv * (h - hpi))
        reg_loss = 0.5 * config.alpha * jnp.mean(h**2 + hpi**2)
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
    data,
    config,
    key,
    steps,
):
    data = reshape_transition(data)

    # B X 1 X obs_dim
    curr_obs = jnp.expand_dims(data.observation[:, 0], axis=1)
    curr_act = jnp.expand_dims(data.action[:, 0], axis=1)
    # B X H+1 X obs/act_dim
    obs_seq = jnp.concatenate([curr_obs, data.next_observation], axis=1)
    init_act_seq = jnp.concatenate([curr_act, data.action], axis=1)

    # B X H+1 X act_dim
    pi_act_seq, _ = policy_model(obs_seq, init_act_seq, key)

    value_loss, value_grads, *priority_aux = value_loss_grad_fun(
        target_value_model=value_model_target,
        value_model=value_model,
        data=data,
        pi_act_seq=pi_act_seq,
        config=config,
    )
    value_optimizer.update(value_grads)

    policy_loss, policy_grads, *policy_aux = policy_loss_grad_fun(
        value_model=value_model,
        policy_model=policy_model,
        data=data,
        pi_act_seq=pi_act_seq,
        config=config,
    )
    policy_cond = (steps % config.policy_update_freq) == 0
    policy_grads = jax.tree.map(
        lambda g: jnp.where(policy_cond, g, jnp.zeros_like(g)),
        policy_grads,
    )
    policy_optimizer.update(policy_grads)

    value_model_target = polyak_update(
        value_model_target, value_model, config.update_tau
    )

    return *priority_aux, (value_loss, policy_cond * policy_loss, *policy_aux)


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
        key, env_state_val, buffer_state, models, val = carry

        env_state, running_state = env_state_val
        policy_model = models[0]

        key, buffer_key = jax.random.split(key)
        env_state, buffer_state = get_experience(
            key=buffer_key,
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            policy=policy_model,
            buffer=buffer,
        )
        running_state = RunningStatistics.insert_reward(running_state, env_state.reward)

        def do_train(j, carry):
            key, env_state_val, buffer_state, models, _ = carry

            (
                policy_model,
                policy_optimizer,
                value_model_target,
                value_model,
                value_optimizer,
            ) = models

            buffer_state, batch_data, idxs = buffer.sample(buffer_state)

            key, train_key = jax.random.split(key)
            steps = config.train_per_step * i + j
            priority, val = train_step(
                value_model_target=value_model_target,
                value_model=value_model,
                value_optimizer=value_optimizer,
                policy_model=policy_model,
                policy_optimizer=policy_optimizer,
                data=batch_data,
                config=config,
                key=train_key,
                steps=steps,
            )
            buffer_state = buffer.update_priorities(buffer_state, idxs, priority)

            carry = (
                key,
                env_state_val,
                buffer_state,
                (
                    policy_model,
                    policy_optimizer,
                    value_model_target,
                    value_model,
                    value_optimizer,
                ),
                val,
            )
            return carry

        init_carry = (key, (env_state, running_state), buffer_state, models, val)
        carry = nnx.fori_loop(1, config.train_per_step + 1, do_train, init_carry)

        return carry

    init_val = (jnp.zeros((), dtype=jnp.float32),) * 6
    init_carry = (
        key,
        (
            env_state,
            running_state,
        ),
        buffer_state,
        (
            policy_model,
            policy_optimizer,
            value_model_target,
            value_model,
            value_optimizer,
        ),
        init_val,
    )
    key, env_state_val, buffer_state, models, val = nnx.fori_loop(
        0, num_steps, body_fun, init_carry
    )

    return *val, *env_state_val, buffer_state, num_steps


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
    config["train_horizon"] = args.train_horizon
    config["alpha"] = args.alpha
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
    prng_key, env_key = jax.random.split(prng_key)
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

    buffer = TrajectorySamplingQueue(
        max_replay_size=config["max_replay_size"],
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size,
        horizon=config["train_horizon"],
    )
    prng_key, buffer_key = jax.random.split(prng_key)
    buffer_state = buffer.init(buffer_key)

    prng_key, running_key = jax.random.split(prng_key)
    running_state = RunningStatistics.init(
        (config["eval_episode_freq"] * config["episode_length"],), running_key
    )

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
            running_state,
            buffer_state,
            num_steps,
        ) = val

        steps += num_steps

        logger.logged = False

        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_value", value_loss.item())

        logger.log_tabular("Loss/Loss_policy", policy_loss.item())
        logger.log_tabular("Loss/Loss_policy_pg", policy_pg_loss.item())
        logger.log_tabular("Loss/Loss_policy_reg", policy_reg_loss.item())

        logger.log_tabular("Loss/policy_q_value", policy_qmean.item())
        logger.log_tabular("Loss/policy_v_value", policy_vmean.item())

        logger.log_tabular("Buffer/max_priority", buffer_state.max_priority)

        logger.log_tabular(
            "Norm/value_model",
            get_tree_norm(nnx.state(value_model, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/policy_model",
            get_tree_norm(nnx.state(policy_model, nnx.Param)),
        )

        logger.log_tabular(
            "Eval/Return",
            running_state.reward_state.data.sum() / config["eval_episode_freq"],
        )

        logger.dump_tabular()

        if (steps - config["warmup_samples"]) % config["save_freq"] == 0:
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
