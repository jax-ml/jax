from jax._src.numpy import lax_numpy as jnp

def l2_loss(pred, targets=None):
    '''L2 (mean squared) loss.
    
    Args:
      pred: float produced by model
      targets: ground-truth float value

    Returns:
      L2 loss (float)  
    '''

    diff = pred - targets if targets is not None else pred
    diff = jnp.square(diff)

    return jnp.mean(diff)

def l1_loss(pred, targets=None):
    '''L1 (absolute) loss.
    
    Args:
      pred: float produced by model
      targets: ground-truth float value

    Returns:
      L1 loss (float)  
    '''

    diff = pred - targets if targets is not None else pred
    diff = jnp.abs(diff)

    return jnp.mean(diff)