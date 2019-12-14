import numpy as onp
import jax.numpy as np
from jax import grad

## SHORT FUNCTION
def f_short(x):
    return x**2

def time_short_fun():
    f_short(2.)

def time_short_grad():
    grad(f_short)(2.)

## LONG FUNCTION
def f_long(x):
    for i in range(50):
        x = np.sin(x)
    return x

def time_long_fun():
    f_long(2.)

def time_long_grad():
    grad(f_long)(2.)

## 'PEARLMUTTER TEST' FUNCTION
def fan_out_fan_in(x):
    for i in range(10**4):
        x = (x + x)/2.0
    return np.sum(x)

def time_fan_out_fan_in_fun():
    fan_out_fan_in(2.)

def time_fan_out_fan_in_grad():
    grad(fan_out_fan_in)(2.)

## UNIT BENCHMARKS
def time_exp_call():
    onp.exp(2.)

def time_exp_primitive_call_unboxed():
    np.exp(2.)

def time_no_autograd_control():
    # Test whether the benchmarking machine is running slowly independent of autograd
    A = np.random.randn(200, 200)
    np.dot(A, A)
