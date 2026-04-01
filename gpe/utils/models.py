import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

EPS = 1e-7


def l2_normalize(x, axis=None, eps=EPS):
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def log1pexp(x, eps=EPS):
    # safe implementation of L = log(1 + exp(x))
    # x > 0: L = x + log(1 + exp(-x))
    # x <=0: L = log(1 + exp(x))
    # combined: L = Relu(x) + log(1 + exp(-|x|)) = jax.nn.softplus(x)
    abs_x = jnp.abs(x)
    pos_x = jax.nn.relu(x)
    return pos_x + jnp.log(1 + jnp.exp(-abs_x))


def get_tree_norm(tree):
    square_tree = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), tree)
    total_square = jax.tree_util.tree_reduce(lambda acc, x: acc + x, square_tree)
    l2_norm = jnp.sqrt(total_square)
    return l2_norm


def bce_loss(logits, labels, weights=1.0):
    """
    Numerically Stable BCE loss
    Doc: https://medium.com/@sahilcarterr/why-nn-bcewithlogitsloss-numerically-stable-6a04f3052967
    """
    tn = jnp.clip(-logits, min=0.0)
    loss = (1 - labels) * logits + tn + jnp.logaddexp(-tn, -logits - tn)
    loss = weights * loss
    return jnp.mean(loss)


class Scalar(nnx.Module):
    def __init__(self, val):
        dtype = jnp.float32
        self.val = nnx.Param(jnp.array(val, dtype=dtype))

    def __call__(self):
        return self.val


class TdmpcValue(nnx.Module):
    def __init__(self, rngs, x_dim, hidden_size=256):
        zero_init = nnx.initializers.zeros

        self.model = nnx.Sequential(
            nnx.Linear(x_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(self, x):
        return jnp.squeeze(self.model(x), axis=-1)


class EnsembleValue(nnx.Module):
    def __init__(self, rngs, x_dim, hidden_size=256):
        self.v1 = TdmpcValue(rngs, x_dim, hidden_size)
        self.v2 = TdmpcValue(rngs, x_dim, hidden_size)

    def __call__(self, x):
        return self.v1(x), self.v2(x)


class PreferencePolicy(nnx.Module):
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        beta=1.0,
        hidden_size=256,
        clip_range=(-0.9, 0.9),
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.beta = beta
        self.clip_range = clip_range
        self._pref_model = TdmpcValue(rngs, obs_dim + act_dim, hidden_size)

    def h(self, obs, act):
        return self._pref_model(jnp.concatenate([obs, act], axis=-1))

    def sampling(self, obs, init_act, key, **kwargs):
        raise Exception("Please implement sampling strategy.")

    def __call__(self, obs, init_act, key, **kwargs):
        return self.sampling(obs, init_act, key, **kwargs)


class MHPolicy(PreferencePolicy):
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        beta=1.0,
        hidden_size=256,
        clip_range=(-0.9, 0.9),
        sigma=1.0,
        decay=0.9,
        num_itr=10,
        num_particles=5,
    ):
        super().__init__(rngs, obs_dim, act_dim, beta, hidden_size, clip_range)
        self.sigma = sigma
        self.decay = decay
        self.num_itr = num_itr
        self.num_particles = num_particles

    def mh_sampling(self, obs, init_act, key):
        # obs:  B X H X obs_dim
        # init_act: B X H X act_dim
        u = init_act
        batch, horizon, act_dim = u.shape
        obs_dim = obs.shape[-1]

        sigma, decay = self.sigma, self.decay
        num_particles = self.num_particles
        u_min, u_max = self.clip_range

        # sample particles
        key, subkey = jax.random.split(key)
        noise = sigma * jax.random.normal(
            subkey, shape=(batch, horizon, num_particles, act_dim)
        )
        # B X H X 1 X act_dim
        u_expand = jnp.expand_dims(u, axis=2)
        # B X H X num_particles X act_dim
        u = u_expand + noise
        # B X H X (num_particles + 1) X act_dim
        u = jnp.concat([u_expand, u], axis=2)
        u = jnp.clip(u, min=u_min, max=u_max)

        # broadcast obs dims
        obs = jnp.expand_dims(obs, axis=2)
        obs = jnp.broadcast_to(obs, (batch, horizon, num_particles + 1, obs_dim))

        def body(i, carry):
            u, sigma, key = carry
            key, subkey1, subkey2 = jax.random.split(key, 3)
            # sample new action
            u_new = u + sigma * jax.random.normal(subkey1, shape=u.shape, dtype=u.dtype)
            u_new = jnp.clip(u_new, min=u_min, max=u_max)
            # estimate adv = min(1.0, exp(h(s, u_new) / beta) / exp(h(s, u) / beta))
            # B X H X (num_particles + 1)
            h_diff = jnp.clip(
                (self.h(obs, u_new) - self.h(obs, u)) / self.beta, max=1.0
            )
            adv = jnp.minimum(1.0, jnp.exp(h_diff))
            # sample action wrt adv
            rand = jax.random.uniform(subkey2, shape=adv.shape)
            # B X H X (num_particles + 1) X 1
            select = jnp.expand_dims(rand < adv, -1)
            # B X H X (num_particles + 1) X act_dim
            u = jnp.where(select, u_new, u)
            return (u, decay * sigma, key)

        init_carry = (u, sigma, key)
        # u: B X H X (num_particles + 1) X act_dim
        (u, sigma, _) = nnx.fori_loop(0, self.num_itr, body, init_carry)

        # h: B X H X (num_particles + 1)
        # max_idx: B X H
        max_idx = jnp.argmax(self.h(obs, u), axis=-1)
        # u: B X H X act_dim
        u = u[
            jnp.arange(batch)[:, None],  # B X 1
            jnp.arange(horizon)[None, :],  # 1 X H
            max_idx,
        ]
        return u, sigma

    def sampling(self, obs, init_act, key, **kwargs):
        return self.mh_sampling(obs, init_act, key, **kwargs)
