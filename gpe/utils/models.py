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
