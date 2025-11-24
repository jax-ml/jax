---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "29WqUVkCXjDD"}

## Practical examples: Control autodiff's saved values with `jax.checkpoint` (aka `jax.remat`)

<!--* freshness: { reviewed: '2024-11-24' } *-->

This notebook provides practical, interactive examples of using `jax.checkpoint` (also called `jax.remat`) to control which intermediate values are saved during automatic differentiation.

**For comprehensive documentation and theory, see the {ref}`gradient-checkpointing` guide.**

```{code-cell}
import jax
import jax.numpy as jnp
```

+++ {"id": "qaIsQSh1XoKF"}

### Quick start

Use the `jax.checkpoint` decorator (aliased as `jax.remat`) with `jax.grad` to control which intermediates are saved on the forward pass versus recomputed on the backward pass, trading off memory and FLOPs.

Without using `jax.checkpoint`, the forward pass of `jax.grad(f)(x)` saves, for use on the backward pass, the values of Jacobian coefficients and other intermediates. We call these saved values _residuals_. Let's examine this with an example:

```{code-cell}
def g(W, x):
  y = jnp.dot(W, x)
  return jnp.sin(y)

def f(W1, W2, W3, x):
  x = g(W1, x)
  x = g(W2, x)
  x = g(W3, x)
  return x

W1 = jnp.ones((5, 4))
W2 = jnp.ones((6, 5))
W3 = jnp.ones((7, 6))
x = jnp.ones(4)

# Inspect the 'residual' values to be saved on the forward pass
# if we were to evaluate `jax.grad(f)(W1, W2, W3, x)`
from jax.ad_checkpoint import print_saved_residuals
jax.ad_checkpoint.print_saved_residuals(f, W1, W2, W3, x)
```

+++ {"id": "97vvWfI-fSSF"}

By applying `jax.checkpoint` to sub-functions, as a decorator or at specific application sites, we force JAX not to save any of that sub-function's residuals. Instead, only the inputs of a `jax.checkpoint`-decorated function might be saved, and any residuals consumed on the backward pass are re-computed from those inputs as needed:

```{code-cell}
def f2(W1, W2, W3, x):
  x = jax.checkpoint(g)(W1, x)
  x = jax.checkpoint(g)(W2, x)
  x = jax.checkpoint(g)(W3, x)
  return x

jax.ad_checkpoint.print_saved_residuals(f2, W1, W2, W3, x)
```

Here the values of two `sin` applications are saved because they are arguments
in subsequent applications of the `jax.checkpoint`-decorated `g` function, and
inputs to a `jax.checkpoint`-decorated function may be saved. But no values of
`cos` applications are saved.

+++ {"id": "CyRR3mTpjRtl"}

To control which values are saveable without having to edit the definition of the function to be differentiated, you can use a rematerialization _policy_. Here is an example that saves only the results of `dot` operations with no batch dimensions (since they are often FLOP-bound, and hence worth saving rather than recomputing):

```{code-cell}
f3 = jax.checkpoint(f, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
jax.ad_checkpoint.print_saved_residuals(f3, W1, W2, W3, x)
```

+++ {"id": "9fe6W0YxlfKa"}

You can also use policies to refer to intermediate values you name using `jax.ad_checkpoint.checkpoint_name`:

```{code-cell}
from jax.ad_checkpoint import checkpoint_name

def f4(W1, W2, W3, x):
  x = checkpoint_name(g(W1, x), name='a')
  x = checkpoint_name(g(W2, x), name='b')
  x = checkpoint_name(g(W3, x), name='c')
  return x

f4 = jax.checkpoint(f4, policy=jax.checkpoint_policies.save_only_these_names('a'))
jax.ad_checkpoint.print_saved_residuals(f4, W1, W2, W3, x)
```

+++ {"id": "40oy-FbmVkDc"}

When playing around with these toy examples, we can get a closer look at what's going on using the `print_fwd_bwd` utility defined in this notebook:

```{code-cell}
from jax.tree_util import tree_flatten, tree_unflatten

from rich.console import Console
from rich.table import Table
import rich.text

def print_fwd_bwd(f, *args, **kwargs) -> None:
  args, in_tree = tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  fwd = jax.make_jaxpr(lambda *args: jax.vjp(f_, *args))(*args).jaxpr

  y, f_vjp = jax.vjp(f_, *args)
  res, in_tree = tree_flatten(f_vjp)

  def g_(*args):
    *res, y = args
    f_vjp = tree_unflatten(in_tree, res)
    return f_vjp(y)

  bwd = jax.make_jaxpr(g_)(*res, y).jaxpr

  table = Table(show_header=False, show_lines=True, padding=(1, 2, 0, 2), box=None)
  table.add_row("[bold green]forward computation:",
                "[bold green]backward computation:")
  table.add_row(rich.text.Text.from_ansi(str(fwd)),
                rich.text.Text.from_ansi(str(bwd)))
  console = Console(width=240, force_jupyter=True)
  console.print(table)

def _renderable_repr(self):
  return self.html
rich.jupyter.JupyterRenderable._repr_html_ = _renderable_repr
```

```{code-cell}
# no use of jax.checkpoint:
print_fwd_bwd(f, W1, W2, W3, x)
```

```{code-cell}
# using jax.checkpoint with policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable:
print_fwd_bwd(f3, W1, W2, W3, x)
```

+++ {"id": "UsvnQJYomcub"}

### Detailed explanation

For a comprehensive walkthrough of the concepts, theory, and best practices, please refer to the {ref}`gradient-checkpointing` guide which includes:

* **Fundamentals of jax.checkpoint** - Understanding memory vs. FLOP tradeoffs
* **Custom policies** - Fine-grained control over what gets saved
* **Recursive checkpointing** - Advanced memory optimization techniques
* **Practical notes** - Integration with `jax.jit` and `jax.lax.scan`

The following sections demonstrate these concepts interactively with visualizations.
