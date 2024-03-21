import jax
from jax import lax

#       def _cond(_):
#         return pl.atomic_cas(lock_ref, 0, 1) == 1
#       lax.while_loop(_cond, lambda a: a, 0)
#       counter_ref[...] += 1
#       pl.atomic_xchg(lock_ref, (), 0)
# def while_loop(cond_fun, body_fun, init_val):
#   val = init_val
#   while cond_fun(val):
#     val = body_fun(val)
#   return val
    # device = xm.xla_device()

def cond_fn(init, limit_value):
    return lax.lt(limit_value[0], init[0]) # limit_value[0] >= init[0]

def body_fn(init, limit_value):
    #   one_value = torch.ones(1, dtype=torch.int32, device=device)
    return (lax.add(init, lax._const(init, 1)), limit_value)

    # TODO(@manfei): init and limit_value has to be torch.tensor.
    # init = torch.tensor([0], dtype=torch.int32, device=device)
    # limit_value = torch.tensor([10], dtype=torch.int32, device=device)
res = lax.while_loop(cond_fn, body_fn, (0, 30))
    # expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    # self.assertEqual(expected, res)
print("res: ", res)

    # xm.mark_step()
    # device = xm.xla_device()

    # # TODO(@manfei): lower, upper and init_val has to be torch.tensor.
    # init_val = torch.tensor([1], dtype=torch.int32, device=device)
    # lower = torch.tensor([0], dtype=torch.int32, device=device)
    # upper = torch.tensor([30], dtype=torch.int32, device=device)
    # one_value = torch.tensor([1], dtype=torch.int32, device=device)
    # init_val_list = (init_val, one_value)
    # # lowers = torch.tensor(([1], [1], [1]), dtype=torch.int32, device=device) # lower, init_val, one_value

    # def body_fun(a, b):
    #   return torch.add(a, b) # [0])
    # # _, _, res, _ = fori_loop(lower, upper, body_fun, init_val, one_value) # init_val_list) # init_val)
    # # A, B, res, D = fori_loop(lower, upper, body_fun, init_val, one_value) # init_val_list) # init_val)
    # # A, B, res, D = fori_loop(upper, body_fun, lowers) # lower, upper, body_fun, init_val, one_value)
    # res, _ = fori_loop(lower, upper, body_fun, init_val, one_value)
    # print("result: ", res) # init_val_