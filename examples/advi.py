"""Automatic differentiation variational inference in Numpy and JAX.

This demo fits a Gaussian approximation to an intractable, unnormalized
density, by differentiating through a Monte Carlo estimate of the
variational evidence lower bound (ELBO)."""

from __future__ import absolute_import
from __future__ import print_function

from functools import partial
import matplotlib.pyplot as plt

from jax.api import jit, grad, vmap
from jax import random
from jax.experimental import minmax
import jax.numpy as np
import jax.scipy.stats.norm as norm


# ========= Functions to define the evidence lower bound. =========

def diag_gaussian_sample(mean, log_std, rng):
    return mean + np.exp(log_std) * random.normal(rng, mean.shape)

def diag_gaussian_logpdf(x, mean, log_std):
    return np.sum(vmap(norm.logpdf)(x, mean, np.exp(log_std)))

def elbo((mean, log_std), logprob, rng):
    # Simple Monte Carlo estimate of the variational lower bound.
    sample = diag_gaussian_sample(mean, log_std, rng)
    return logprob(sample) - diag_gaussian_logpdf(sample, mean, log_std)


# ========= Helper functions for batching. =========

def rng_map(f, rng, num_samples, *args, **kwargs):
    # Calls f with a batch of different seeds.
    # f must have signature f(*args, rng, **kwargs).
    rngs = np.array(random.split(rng, num_samples))
    return vmap(partial(f, *args, **kwargs))(rngs)

def batch_elbo(params, logprob, num_samples, rng):
    return np.mean(rng_map(elbo, rng, num_samples, params, logprob))


# ========= Helper functions for plotting. =========

def eval_zip(func, X, Y):
    params_vec = np.stack([X.ravel(), Y.ravel()]).T
    zs = vmap(func)(params_vec)
    return zs.reshape(X.shape)
eval_zip = jit(eval_zip, static_argnums=(0,))

def plot_isocontours(ax, func, xlimits, ylimits, numticks=101, **kwargs):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    Z = eval_zip(func, X, Y)
    ax.contour(X, Y, Z, **kwargs)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_yticks([])
    ax.set_xticks([])


# ========= Define an intractable unnormalized density =========

def funnel_log_density(params):
    mean, log_stddev = params[0], params[1]
    return norm.logpdf(mean,        0, np.exp(log_stddev)) + \
           norm.logpdf(log_stddev,  0, 1.35)


if __name__ == "__main__":
    step_size = 0.1
    momentum = 0.9
    num_steps = 100
    num_samples = 20

    def objective(params, rng):
        return -batch_elbo(params, funnel_log_density, num_samples, rng)

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    xlimits = [-2, 2]
    ylimits = [-4, 2]

    def callback(params, t, rng):
        print("Iteration {} lower bound {}".format(t, objective(params, rng)))

        plt.cla()
        target_dist = lambda x: np.exp(funnel_log_density(x))
        approx_dist = lambda x: np.exp(diag_gaussian_logpdf(x, *params))
        plot_isocontours(ax, target_dist, xlimits, ylimits, cmap='summer')
        plot_isocontours(ax, approx_dist, xlimits, ylimits, cmap='winter')

        samples = rng_map(diag_gaussian_sample, rng, num_samples, *params)
        plt.plot(samples[:, 0], samples[:, 1], 'b.')
        plt.draw()
        plt.pause(1.0/60.0)


    # Set up optimizer.
    D = 2
    init_mean = np.zeros(D)
    init_std  = np.zeros(D)
    init_params = (init_mean, init_std)
    opt_init, opt_update = minmax.momentum(step_size, mass=momentum)
    opt_state = opt_init(init_params)

    @jit
    def update(i, opt_state, rng):
        params = minmax.get_params(opt_state)
        return opt_update(i, grad(objective)(params, rng), opt_state)


    # Main fitting loop.
    print("Optimizing variational parameters...")
    rng = random.PRNGKey(0)
    for t in range(num_steps):
      opt_state = update(t, opt_state, rng)
      callback(minmax.get_params(opt_state), t, rng)
      old, rng = random.split(rng)
