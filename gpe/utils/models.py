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


class ExpCostModel(nnx.Module):
    def __init__(self, rngs, x_dims, hidden_size=256, clip_range=(0.0, 1.0)):
        sizes = [x_dims, hidden_size, hidden_size, 1]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nnx.elu if j < len(sizes) - 2 else jax.nn.identity
            affine_layer = nnx.Linear(sizes[j], sizes[j + 1], rngs=rngs)
            layers += [affine_layer, act]
        self.model = nnx.Sequential(*layers)
        self.min, self.max = clip_range

    def __call__(self, x):
        x = jnp.squeeze(self.model(x), axis=-1)
        ret = jax.nn.sigmoid(x)
        ret = jnp.clip(ret, min=self.min, max=self.max)
        return ret


class ContrastiveCostModel(nnx.Module):
    def __init__(self, rngs, x_dims, hidden_size=256):
        sizes = [x_dims, hidden_size, hidden_size, 128]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nnx.elu if j < len(sizes) - 2 else jax.nn.identity
            affine_layer = nnx.Linear(sizes[j], sizes[j + 1], rngs=rngs)
            layers += [affine_layer, act]
        self.encoder = nnx.Sequential(*layers)
        self.projection = nnx.Linear(sizes[-1], 1, rngs=rngs)

    def __call__(self, x):
        z = l2_normalize(self.encoder(x), axis=-1)
        proj_z = jnp.squeeze(self.projection(z), axis=-1)
        cost = jax.nn.sigmoid(proj_z)
        return z, cost


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


def mh_sampling(key, model, obs, init_act, beta=1.0, num_itr=10, clip_range=(-0.9, 0.9)):
    u_min, u_max = clip_range
    
    def sample(i, carry):
        key, u = carry
        key, subkey1, subkey2 = jax.random.split(key, 3)
        # sample new action
        u_new = u + 0.3*jax.random.normal(subkey1, shape=u.shape, dtype=u.dtype)
        u_new = jnp.clip(u_new, min=u_min, max=u_max)
        # estimate adv
        adv = jnp.exp((model(obs, u_new) - model(obs, u)) / beta)
        adv = jnp.minimum(1.0, adv)
        # sample action wrt adv
        rand = jax.random.uniform(subkey2, shape=adv.shape)
        u = jnp.where(rand < adv, u_new, u)
        return (key, u)
    
    init_carry = (key, init_act)
    (_, u) = jax.lax.fori_loop(0, num_itr, sample, init_carry)
    return u


class MHPolicy(nnx.Module):
    def __init__(self, rngs, obs_dim, act_dim, beta, hidden_size=256):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.beta = beta
        self._pref_model = TdmpcValue(rngs, obs_dim + act_dim, hidden_size)

    def h(self, obs, act):
        return self._pref_model(jnp.concatenate([obs, act], axis=-1))

    def action(self, key, obs, init_act, num_itr=10):
        return mh_sampling(
            key=key,
            model=self._pref_model,
            obs=obs,
            init_act=init_act,
            beta=self.beta,
            num_itr=num_itr
        )

    def __call__(self, obs, init_act, key):
        return self.action(key, obs, init_act)
 