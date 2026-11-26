import jax
import jax.numpy as jnp
from typing import Callable, Iterable
from newton_method import wolfe, g, ini

def h2h(hessian):
    l = hessian.shape[0] # hessian.shape = (l, l)
    while 1:
        rank = jnp.linalg.matrix_rank(hessian)
        if rank == l:
            break
        else:
            hessian += jnp.identity(l)
    return hessian

def bfgs(fun: Callable, 
         x_0: Iterable,
         verbose: bool=False,
         epsilon: float=1e-6, 
         k: int=0):
    hessian = jax.hessian(fun)(x_0)
    hessian = h2h(hessian)
    while 1:
        gradient = jax.grad(fun)(x_0)
        dk = -jax.numpy.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if verbose:
            print('{}\t{}\t{}'.format(x_0, fun(x_0), k))
        if jax.numpy.linalg.norm(dk) >= epsilon:
            alpha = wolfe(fun, gradient, x_0, dk)
            delta = alpha * dk # sk, same as learning_date * grad(corrected by hessian and jax.grad)
            x_0 += delta[0]
            yk = jax.grad(fun)(x_0) - gradient
            s = delta.reshape(-1)                    
            sTy = float(s.dot(yk))                   
            if jnp.all(jnp.isfinite(yk)) and jnp.isfinite(sTy) and sTy > 1e-12:
                Hs = hessian.dot(delta.T)
                sTHs = float(s.dot(Hs.reshape(-1)))
                if jnp.isfinite(sTHs) and sTHs > 1e-12:
                    hessian += jnp.outer(yk, yk) / sTy - Hs.dot(Hs.T) / sTHs
            else:
                hessian = h2h(jnp.eye(hessian.shape[0], dtype=hessian.dtype))
            k += 1
        else:
            break
    return x_0

if __name__ =='__main__':
    print(bfgs(g, ini, True))