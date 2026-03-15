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
from gpe.utils.buffer import UniformSamplingQueue
from gpe.utils.logger import EpochLogger
from gpe.utils.models import EnsembleValue, MHPolicy, get_tree_norm
from gpe.utils.types import Transition
from gpe.utils.utils import make_static_config_from_dict, single_agent_args

jax.config.update("jax_disable_jit", True)
EPS = 1e-6

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "max_grad_norm": 5.0,
    "gamma": 0.99,
    "update_tau": 0.005,
    "weight_decay": 0.01,
    "episode_length": 1000,
    "warmup_samples": int(10),
    "max_replay_size": int(1e5),
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


@functools.partial(nnx.jit, static_argnames="buffer")
def prefill_buffer(
    key,
    env,
    env_state,
    buffer_state,
    policy,
    buffer,
    num_itr=1000,
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

    (_, env_state, buffer_state), () = jax.lax.scan(
        body,
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


def compute_target_value(cost_model, bc_policy, value_model, obs, act, done, gamma):
    dtype = obs.dtype

    # Batch X Horizon X obs_dim
    batch, horizon, _ = obs.shape

    # Batch X Horizon X act_dim
    act_next, *_ = bc_policy(obs.reshape(batch * horizon, -1))
    act_next = act_next.reshape(batch, horizon, -1)

    # mask last horizon
    mask_last_horizon = jnp.concat(
        [jnp.ones(horizon - 1, dtype=dtype), jnp.zeros(1, dtype=dtype)]
    )
    # This is a permutation matrix to shift one timestep ahead
    shift_one_timestep = jnp.eye(horizon, k=-1, dtype=dtype)

    def batch_fun(obs, act, act_next, done):
        # Batch X Horizon
        _, cost = cost_model(jnp.concat([obs, act], axis=-1))
        cost = mask_last_horizon * cost
        # Batch X Horizon
        v_next = jnp.maximum(*value_model(jnp.concat([obs, act_next], axis=-1)))
        v_next = v_next @ shift_one_timestep

        target = cost + gamma * (1 - done) * v_next
        return target

    target_value = jax.vmap(batch_fun, in_axes=(0, 0, 0, 0))(obs, act, act_next, done)
    return target_value


def value_loss_grad_fun(
    target_value_model,
    value_model,
    policy_model,
    data,
    gamma,
):
    print(data.observation.shape)
    raise Exception
    dtype = data.union_obs.dtype
    horizon = data.horizon

    # mask the horizon-1 element
    mask = jnp.concat([jnp.ones(horizon - 1, dtype=dtype), jnp.zeros(1, dtype=dtype)])

    # Batch X Horizon
    target_v = compute_target_value(
        cost_model=cost_model,
        bc_policy=bc_policy,
        value_model=target_value_model,
        obs=data.union_obs,
        act=data.union_act,
        done=data.union_done,
        gamma=gamma,
    )

    def loss_fun(value_model):
        # Batch X Horizon X obs_act_dim
        target_oa = jnp.concat([data.union_obs, data.union_act], axis=-1)
        # Batch X Horizon
        pred_v1, pred_v2 = value_model(target_oa)
        v1_loss = mask * optax.huber_loss(pred_v1, target_v, delta=2.0)
        v2_loss = mask * optax.huber_loss(pred_v2, target_v, delta=2.0)
        discounted_v1_loss = discounted_sum(v1_loss.T, gamma)
        discounted_v2_loss = discounted_sum(v2_loss.T, gamma)

        loss = jnp.mean(discounted_v1_loss) + jnp.mean(discounted_v2_loss)
        return loss

    grad_fun = nnx.value_and_grad(loss_fun)
    loss, grads = grad_fun(value_model)
    grads = jax.tree.map(lambda g: g / horizon, grads)

    return loss, grads


def compute_value_weight(
    value_model,
    bc_policy,
    obs,
    act,
    alpha,
    beta,
    use_osil_weight,
):
    pred_act, *_ = bc_policy(obs)

    # Batch_Horizon
    q = jnp.maximum(*value_model(jnp.concat([obs, act], axis=-1)))
    v = jnp.maximum(*value_model(jnp.concat([obs, pred_act], axis=-1)))

    def per_transition_weight():
        neg_adv = jnp.clip(-(q - v) / beta, max=2.0)
        union_weight = jnp.exp(neg_adv) / jnp.mean(jnp.abs(v))
        return union_weight

    def constant_adaptive_weight():
        # osil type union weight
        neg_adv = -(q - v) / beta
        log_Z = jax.nn.logsumexp(neg_adv) - jnp.log(neg_adv.shape[0]) + EPS
        union_weight = alpha * jnp.exp(jnp.clip(log_Z, max=5.0))
        return union_weight

    union_weight = jnp.where(
        use_osil_weight, constant_adaptive_weight(), per_transition_weight()
    )
    return union_weight, jnp.abs(q).mean(), jnp.abs(v).mean()


def policy_loss_grad_fun(
    value_model,
    bc_policy,
    data,
    config,
    has_positive,
):
    # Batch X Horizon X obs_dim
    batch, horizon, _ = data.union_obs.shape

    # Batch_Horizon X obs/act_dim
    target_pos_obs = data.pos_obs.reshape(batch * horizon, -1)
    target_pos_act = data.pos_act.reshape(batch * horizon, -1)

    target_union_obs = data.union_obs.reshape(batch * horizon, -1)
    target_union_act = data.union_act.reshape(batch * horizon, -1)

    union_weight, qmean, vmean = compute_value_weight(
        value_model=value_model,
        bc_policy=bc_policy,
        obs=target_union_obs,
        act=target_union_act,
        alpha=config.alpha,
        beta=config.beta,
        use_osil_weight=config.use_osil_weight,
    )

    def loss_fun(bc_policy):
        pred_pos_act, *_ = bc_policy(target_pos_obs)
        pos_loss = optax.l2_loss(pred_pos_act, target_pos_act).sum(axis=-1)
        pos_bc_loss = has_positive * jnp.mean(pos_loss)

        pred_union_act, *_ = bc_policy(target_union_obs)
        union_loss = optax.l2_loss(pred_union_act, target_union_act).sum(axis=-1)
        union_bc_loss = jnp.mean(union_loss)

        union_value = jnp.maximum(
            *value_model(jnp.concat([target_union_obs, pred_union_act], axis=-1))
        )
        union_value_loss = jnp.mean(union_weight * union_value)

        loss = pos_bc_loss + union_bc_loss + union_value_loss
        return loss, (pos_bc_loss, union_bc_loss, union_value_loss)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(bc_policy)

    return loss, grads, *aux_values, qmean, vmean


def train_step(
    value_model_target,
    value_model,
    value_optimizer,
    policy_model,
    policy_optimizer,
    batch_data,
    config,
    steps,
):
    value_loss, value_grads, *value_aux = value_loss_grad_fun(
        target_value_model=value_model_target,
        value_model=value_model,
        policy_model=policy_model,
        data=batch_data,
        gamma=config.gamma,
    )
    value_optimizer.update(value_grads)

    policy_loss, policy_grads, *policy_aux = policy_loss_grad_fun(
        value_model=value_model,
        policy_model=policy_model,
        data=batch_data,
        config=config,
    )
    policy_optimizer.update(policy_grads)

    value_model_target = polyak_update(
        value_model_target, value_model, config.update_tau
    )

    return (
        value_loss,
        *value_aux,
        policy_loss,
        *policy_aux,
    )


@functools.partial(nnx.jit, static_argnames="buffer")
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
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
            key,
            env_state,
            buffer_state,
            value_model_target,
            value_model,
            value_optimizer,
            policy_model,
            policy_optimizer,
        ) = carry

        key, subkey = jax.random.split(key)

        env_state, buffer_state = get_experience(
            key=subkey,
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            policy=policy_model,
            buffer=buffer,
        )

        buffer_state, batch_data = buffer.sample(buffer_state)

        val = train_step(
            value_model_target=value_model_target,
            value_model=value_model,
            value_optimizer=value_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            batch_data=batch_data,
            config=config,
            steps=i,
        )

        return (
            val,
            key,
            env_state,
            buffer_state,
            value_model_target,
            value_model,
            value_optimizer,
            policy_model,
            policy_optimizer,
        )

    init_val = (jnp.zeros((), dtype=jnp.float32),) * 19
    init_carry = (
        init_val,
        key,
        env_state,
        buffer_state,
        value_model_target,
        value_model,
        value_optimizer,
        policy_model,
        policy_optimizer,
    )
    val, *_ = nnx.fori_loop(0, num_steps, body_fun, init_carry)

    return val, num_steps


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
    config["alpha"] = args.alpha
    config["beta"] = args.beta
    config["lmbda"] = args.lmbda
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation
    config["lr"] = args.lr

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")
    config["batch_size"] = batch_size

    config_data = make_static_config_from_dict(name="State", d=config)()

    # environment
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

    dummy_obs = jnp.zeros((obs_dim,))
    dummy_action = jnp.zeros((act_dim,))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": 0.0}, "policy_extras": {}},
    )

    buffer = UniformSamplingQueue(
        max_replay_size=config["max_replay_size"],
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size,
    )
    prng_key, buffer_key = jax.random.split(prng_key)
    buffer_state = buffer.init(buffer_key)

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

    logger.log("")
    steps = buffer.size(buffer_state)
    while steps < config["total_iteration"]:
        prng_key, subkey = jax.random.split(prng_key)

        val, num_itr = train_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            value_model_target=value_model_target,
            value_model=value_model,
            value_optimizer=value_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            config=config_data,
            key=subkey,
        )

        (
            cost_loss,
            pos_neg_pref_loss,
            pos_union_pref_loss,
            union_neg_pref_loss,
            pref_loss,
            contrastive_loss,
            value_loss,
            policy_loss,
            policy_pos_bc_loss,
            policy_union_bc_loss,
            policy_union_value_loss,
            policy_qmean,
            policy_vmean,
            mean_pos_reward,
            mean_neg_reward,
            mean_union_reward,
            mean_pos_cost,
            mean_neg_cost,
            mean_union_cost,
        ) = val

        steps += num_itr

        logger.logged = False

        if (steps % config["log_freq"] == 0) and (not logger.logged):

            logger.log_tabular("Train/Steps", steps)

            logger.log_tabular("Loss/Loss_cost", cost_loss.item())
            logger.log_tabular("Loss/Loss_pos_neg_pref_cost", pos_neg_pref_loss.item())
            logger.log_tabular(
                "Loss/Loss_pos_union_pref_cost", pos_union_pref_loss.item()
            )
            logger.log_tabular(
                "Loss/Loss_union_neg_pref_cost", union_neg_pref_loss.item()
            )
            logger.log_tabular("Loss/Loss_pref_cost", pref_loss.item())
            logger.log_tabular("Loss/Loss_contrastive_cost", contrastive_loss.item())

            logger.log_tabular("Loss/Loss_value", value_loss.item())

            logger.log_tabular("Loss/Loss_policy", policy_loss.item())
            logger.log_tabular("Loss/Loss_policy_pos_bc", policy_pos_bc_loss.item())
            logger.log_tabular("Loss/Loss_policy_union_bc", policy_union_bc_loss.item())
            logger.log_tabular(
                "Loss/Loss_policy_union_value", policy_union_value_loss.item()
            )
            logger.log_tabular("Loss/policy_q_value", policy_qmean.item())
            logger.log_tabular("Loss/policy_v_value", policy_vmean.item())

            logger.log_tabular("Reward/pos", mean_pos_reward.item())
            logger.log_tabular("Reward/neg", mean_neg_reward.item())
            logger.log_tabular("Reward/union", mean_union_reward.item())

            logger.log_tabular("Cost/pos", mean_pos_cost.item())
            logger.log_tabular("Cost/neg", mean_neg_cost.item())
            logger.log_tabular("Cost/union", mean_union_cost.item())

            logger.log_tabular(
                "Norm/value_model",
                get_tree_norm(nnx.state(value_model, nnx.Param)),
            )
            logger.log_tabular(
                "Norm/policy_model",
                get_tree_norm(nnx.state(policy_model, nnx.Param)),
            )
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
