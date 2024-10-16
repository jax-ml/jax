# Advanced implementation of JAX's JVP and VJP computations with optimizations.

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import functools
import itertools as it
from functools import partial
from typing import Any, Dict, List, Tuple

from jax._src import config
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax.tree_util import (tree_flatten, tree_unflatten,
                           register_pytree_node, Partial)
from jax._src import core
from jax._src import source_info_util
from jax._src.ad_util import (
    add_jaxvals, replace_internal_symbolic_zeros, zeros_like_jaxval,
    replace_rule_output_symbolic_zeros, Zero, zeros_like_aval)
from jax._src.ad_util import zeros_like_p, add_jaxvals_p  # noqa: F401
from jax._src.api_util import flatten_fun, flatten_fun_nokwargs
from jax._src.core import (Trace, Tracer, get_aval, call_p, Primitive, Literal,
                           raise_to_shaped)
from jax._src.dtypes import dtype, float0
from jax._src.util import (unzip2, safe_map, safe_zip, split_list, wrap_name,
                           as_hashable_function, weakref_lru_cache,
                           partition_list)

# Safe versions of map and zip that enforce equal lengths.
zip = safe_zip
map = safe_map

def identity(x): return x

def _update_annotation(
    f: lu.WrappedFun,
    orig_type: tuple[tuple[core.AbstractValue, bool], ...] | None,
    explicit_nonzeros: List[bool]
) -> lu.WrappedFun:
    if orig_type is None:
        return f
    # By convention, `explicit_nonzeros` only accounts for explicit arguments.
    assert len(explicit_nonzeros) == sum(explicit for _, explicit in orig_type)
    # Implicit arguments never have tangents, so generate the tangent part of the
    # type annotation from explicit arguments only.
    explicit_avals = [aval for aval, explicit in orig_type if explicit]
    tan_types = [(aval.to_tangent_aval(), True)
                 for nz, aval in zip(explicit_nonzeros, explicit_avals) if nz]
    return lu.annotate(f, (*orig_type, *tan_types))

def jvp(fun: lu.WrappedFun, has_aux=False, instantiate=True,
        transform_stack=True) -> Any:
    if not has_aux:
        return jvpfun(jvp_subtrace(fun), instantiate, transform_stack)
    else:
        fun, aux = jvp_subtrace_aux(fun)
        return jvpfun(fun, instantiate, transform_stack), aux

@lu.transformation
def jvpfun(instantiate, transform_stack, primals, tangents):
    # Efficient handling of float0 tangents.
    tangents = [Zero.from_primal_value(t) if not isinstance(t, Zero)
                and dtype(t) == float0 else t for t in tangents]
    ctx = (source_info_util.transform_name_stack('jvp') if transform_stack
           else contextlib.nullcontext())
    with core.new_main(JVPTrace) as main, ctx:
        out_primals, out_tangents = yield (main, primals, tangents), {}
        del main
    if isinstance(instantiate, bool):
        instantiate = [instantiate] * len(out_tangents)
    out_tangents = [instantiate_zeros(t) if inst else t for t, inst
                    in zip(out_tangents, instantiate)]
    yield out_primals, out_tangents

@lu.transformation
def jvp_subtrace(main, primals, tangents):
    trace = JVPTrace(main, core.cur_sublevel())
    for x in list(primals) + list(tangents):
        if isinstance(x, Tracer):
            if x._trace.level >= trace.level:
                raise core.escaped_tracer_error(
                    x, f"Tracer from a higher level: {x} in trace {trace}")
            assert x._trace.level < trace.level
    in_tracers = [JVPTracer(trace, x, t) if type(t) is not Zero else x
                  for x, t in zip(primals, tangents)]
    ans = yield in_tracers, {}
    out_tracers = map(trace.full_raise, ans)
    yield unzip2([(out_tracer.primal, out_tracer.tangent)
                  for out_tracer in out_tracers])

@lu.transformation_with_aux
def jvp_subtrace_aux(main, primals, tangents):
    trace = JVPTrace(main, core.cur_sublevel())
    for x in list(primals) + list(tangents):
        if isinstance(x, Tracer):
            assert x._trace.level < trace.level
    ans, aux = yield map(partial(JVPTracer, trace), primals, tangents), {}
    ans_tracers = map(trace.full_raise, ans)
    out_primals, out_tangents = unzip2((t.primal, t.tangent) for t in ans_tracers)
    aux_primals = [core.full_lower(x.primal)
                   if isinstance(x, JVPTracer) and x._trace.level == trace.level
                   else x for x in aux]
    yield (out_primals, out_tangents), aux_primals

def linearize(traceable, *primals, **kwargs):
    has_aux = kwargs.pop('has_aux', False)
    if not has_aux:
        jvpfun = jvp(traceable)
    else:
        jvpfun, aux = jvp(traceable, has_aux=True)

    in_pvals = (tuple(pe.PartialVal.known(p) for p in primals)
                + tuple(pe.PartialVal.unknown(get_aval(p).to_tangent_aval())
                        for p in primals))
    _, in_tree = tree_flatten(((primals, primals), {}))
    jvpfun_flat, out_tree = flatten_fun(jvpfun, in_tree)
    jaxpr, out_pvals, consts = pe.trace_to_jaxpr_nounits(jvpfun_flat, in_pvals)
    out_primals_pvals, out_tangents_pvals = tree_unflatten(out_tree(), out_pvals)
    assert all(out_primal_pval.is_known() for out_primal_pval in out_primals_pvals)
    out_primals_consts = [pval.get_known() for pval in out_primals_pvals]
    if not has_aux:
        return out_primals_consts, out_tangents_pvals, jaxpr, consts
    else:
        return out_primals_consts, out_tangents_pvals, jaxpr, consts, aux()

def backward_pass(jaxpr: core.Jaxpr, transform_stack,
                  consts, primals_in, cotangents_in):
    if all(type(ct) is Zero for ct in cotangents_in) and not jaxpr.effects:
        return [Zero(v.aval) for v in jaxpr.invars]

    # Efficient storage using dictionaries for cotangents and primals.
    ct_env: Dict[Any, Any] = {}
    primal_env: Dict[Any, Any] = {}

    def write_cotangent(prim, v, ct):
        if ct is None or type(v) is Literal:
            return
        if ct is Zero:
            return
        ct_env[v] = add_tangents(ct_env.get(v, Zero(v.aval)), ct)

    def read_cotangent(v):
        return ct_env.pop(v, Zero(v.aval.to_tangent_aval()))

    def read_primal(v):
        if type(v) is Literal:
            return v.val
        else:
            a = v.aval
            if isinstance(a, core.DShapedArray):
                shape = [primal_env[d] if isinstance(d, core.Var) else d for d in a.shape]
                a = a.update(shape=tuple(shape))
            return primal_env.get(v, UndefinedPrimal(a))

    def write_primal(v, val):
        if not is_undefined_primal(val):
            primal_env[v] = val

    map(write_primal, jaxpr.constvars, consts)
    map(write_primal, jaxpr.invars, primals_in)

    ctx = (source_info_util.transform_name_stack('transpose') if transform_stack
           else contextlib.nullcontext())
    with ctx:
        map(partial(write_cotangent, 'outvars'), jaxpr.outvars, cotangents_in)
        for eqn in reversed(jaxpr.eqns):
            invals = map(read_primal, eqn.invars)
            if eqn.primitive.multiple_results:
                cts_in = map(read_cotangent, eqn.outvars)
            else:
                cts_in, = map(read_cotangent, eqn.outvars)
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(
                eqn.source_info.traceback, name_stack=name_stack), eqn.ctx.manager:
                if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
                    cts_in_avals = [v.aval for v in eqn.outvars]
                    params = dict(eqn.params)
                    call_jaxpr = params.pop('call_jaxpr')
                    cts_out = get_primitive_transpose(eqn.primitive)(
                        params, call_jaxpr, invals, cts_in, cts_in_avals)
                elif eqn.primitive in reducing_transposes:
                    cts_out = reducing_transposes[eqn.primitive](
                        cts_in, *invals, **eqn.params)
                else:
                    cts_out = get_primitive_transpose(eqn.primitive)(
                        cts_in, *invals, **eqn.params)
                cts_out = [Zero(v.aval) for v in eqn.invars] if cts_out is Zero else cts_out
                map(partial(write_cotangent, eqn.primitive), eqn.invars, cts_out)

    cotangents_out = map(read_cotangent, jaxpr.invars)
    return cotangents_out

def closed_backward_pass(jaxpr: core.ClosedJaxpr, transform_stack,
                         primals_in, cotangents_in):
    return backward_pass(jaxpr.jaxpr, transform_stack, jaxpr.consts,
                         primals_in, cotangents_in)

class UndefinedPrimal:
    __slots__ = ['aval']
    def __init__(self, aval):
        self.aval = aval
    def __repr__(self):
        return f'UndefinedPrimal({self.aval})'

def is_undefined_primal(x):
    return isinstance(x, UndefinedPrimal)

register_pytree_node(UndefinedPrimal,
                     lambda z: ((), z.aval),
                     lambda aval, _: UndefinedPrimal(aval))

def get_primitive_transpose(p):
    try:
        return primitive_transposes[p]
    except KeyError as err:
        raise NotImplementedError(
            f"Transpose rule (for reverse-mode differentiation) for '{p}' "
            "not implemented") from err

@lu.transformation_with_aux
def nonzero_tangent_outputs(*args, **kwargs):
    results = (_, tangents_out) = yield args, kwargs
    yield results, [not isinstance(r, Zero) for r in tangents_out]

class JVPTrace(Trace):
    def pure(self, val):
        tangent_zero = Zero.from_primal_value(val)
        return JVPTracer(self, val, tangent_zero)

    def lift(self, val):
        tangent_zero = Zero.from_primal_value(val)
        return JVPTracer(self, val, tangent_zero)

    def sublift(self, val):
        return JVPTracer(self, val.primal, val.tangent)

    def process_primitive(self, primitive, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        jvp = primitive_jvps.get(primitive)
        if not jvp:
            msg = f"Differentiation rule for '{primitive}' not implemented"
            raise NotImplementedError(msg)
        primal_out, tangent_out = jvp(primals_in, tangents_in, **params)
        if primitive.multiple_results:
            return [JVPTracer(self, x, t) for x, t in zip(primal_out, tangent_out)]
        else:
            return JVPTracer(self, primal_out, tangent_out)

    def process_call(self, call_primitive, f, tracers, params):
        assert call_primitive.multiple_results
        primals, tangents = unzip2((t.primal, t.tangent) for t in tracers)
        which_nz = [not isinstance(t, Zero) for t in tangents]
        tangents = [t if not isinstance(t, Zero) else None for t in tangents]
        args, in_tree = tree_flatten((primals, tangents))
        f_jvp = jvp_subtrace(f, self.main)
        f_jvp, which_nz_out = nonzero_tangent_outputs(f_jvp)
        if isinstance(call_primitive, core.MapPrimitive):
            in_axes = params['in_axes']
            tangent_in_axes = [ax for ax, nz in zip(in_axes, which_nz) if nz]
            out_axes_thunk = params['out_axes_thunk']
            @as_hashable_function(closure=out_axes_thunk)
            def new_out_axes_thunk():
                out_ax = out_axes_thunk()
                return (*out_ax, *(ax for ax, nz in zip(out_ax, which_nz_out()) if nz))
            params = dict(params, in_axes=(*in_axes, *tangent_in_axes),
                          out_axes_thunk=new_out_axes_thunk)
        f_jvp, out_tree = traceable(f_jvp, in_tree)
        update_params = call_param_updaters.get(call_primitive)
        new_params = update_params(params, which_nz) if update_params else params
        result = call_primitive.bind(_update_annotation(f_jvp, f.in_type, which_nz),
                                     *args, **new_params)
        primal_out, tangent_out = tree_unflatten(out_tree(), result)
        tangent_out = [Zero.from_primal_value(p) if t is None else t
                       for p, t in zip(primal_out, tangent_out)]
        return [JVPTracer(self, p, t) for p, t in zip(primal_out, tangent_out)]

    def post_process_call(self, call_primitive, out_tracers, params):
        primals, tangents = unzip2((t.primal, t.tangent) for t in out_tracers)
        out, treedef = tree_flatten((primals, tangents))
        tangents_nz = [not isinstance(t, Zero) for t in tangents]
        del primals, tangents
        main = self.main

        def todo(x):
            primals, tangents = tree_unflatten(treedef, x)
            trace = JVPTrace(main, core.cur_sublevel())
            return map(partial(JVPTracer, trace), primals, tangents)

        if call_primitive.map_primitive:
            def out_axes_transform(out_axes):
                return (*out_axes, *(ax for ax, nz in zip(out_axes, tangents_nz) if nz))
            todo = (todo, out_axes_transform)
        return out, todo

    process_map = process_call
    post_process_map = post_process_call

    def process_custom_jvp_call(self, _, __, f_jvp, tracers, *, symbolic_zeros):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        primals_in = map(core.full_lower, primals_in)
        if not symbolic_zeros:
            tangents_in = map(instantiate_zeros, tangents_in)
        else:
            tangents_in = map(replace_internal_symbolic_zeros, tangents_in)
        outs = f_jvp.call_wrapped(*it.chain(primals_in, tangents_in))
        primals_out, tangents_out = split_list(outs, [len(outs) // 2])
        tangents_out = map(replace_rule_output_symbolic_zeros, tangents_out)
        return map(partial(JVPTracer, self), primals_out, tangents_out)

    def post_process_custom_jvp_call(self, out_tracers, _):
        raise CustomJVPException()

    def process_custom_vjp_call(self, _, __, fwd, bwd, tracers, out_trees,
                                symbolic_zeros):
        # Local import to prevent an import cycle.
        from jax._src.lax import lax

        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        fwd_in = [(core.full_lower(p), not isinstance(t, Zero))
                  for p, t in zip(primals_in, tangents_in)]
        fwd_in = [x for pair in fwd_in for x in pair]
        res_and_primals_out = fwd.call_wrapped(*fwd_in)
        _, res_tree = out_trees()
        res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
        avals_out = [raise_to_shaped(core.get_aval(x)).to_tangent_aval() for x in primals_out]
        tangents_in = map(instantiate_zeros, tangents_in)
        tangents_out = custom_lin_p.bind(
            *res, *tangents_in, num_res=res_tree.num_leaves, bwd=bwd,
            out_avals=avals_out, symbolic_zeros=symbolic_zeros)
        tangents_out = map(lax.tie_p.bind, primals_out, tangents_out)
        return map(partial(JVPTracer, self), primals_out, tangents_out)

    def post_process_custom_vjp_call(self, out_tracers, _):
        raise CustomVJPException()

    def process_custom_transpose(self, prim, call, tracers, **params):
        ps_in, ts_in = unzip2((t.primal, t.tangent) for t in tracers)
        res_ps_in, lin_ps_in = split_list(ps_in, [params['res_tree'].num_leaves])
        res_ts_in, lin_ts_in = split_list(ts_in, [params['res_tree'].num_leaves])

        if any(not isinstance(t, Zero) for t in res_ts_in):
            raise NotImplementedError(
                'JVP of custom transpose with respect to non-symbolic-zero residuals')

        ps_out = prim.bind(call, *ps_in, **params)

        lin_ts_in = map(instantiate_zeros, lin_ts_in)
        ts_out = prim.bind(call, *res_ps_in, *lin_ts_in, **params)

        return map(partial(JVPTracer, self), ps_out, ts_out)

    def join(self, xt, yt):
        xz, yz = isinstance(xt, Zero), isinstance(yt, Zero)
        if xz == yz:
            return xt, yt
        elif yz and not xz:
            return xt, zeros_like_jaxval(xt)
        elif xz and not yz:
            return zeros_like_jaxval(yt), yt
        else:
            raise TypeError((xt, yt))

class JVPTracer(Tracer):
    __slots__ = ['primal', 'tangent']

    def __init__(self, trace, primal, tangent):
        if config.enable_checks.value:
            _primal_tangent_shapes_match(primal, tangent)
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return get_aval(self.primal)

    def full_lower(self):
        if isinstance(self.tangent, Zero):
            return core.full_lower(self.primal)
        else:
            return self

def _primal_tangent_shapes_match(primal, tangent):
    if not isinstance(tangent, Zero):
        primal_aval = raise_to_shaped(get_aval(primal), weak_type=False)
        tangent_aval = raise_to_shaped(get_aval(tangent), weak_type=False)
        assert core.definitely_equal_shape(primal_aval.shape, tangent_aval.shape)
        expected_tangent_dtype = core.primal_dtype_to_tangent_dtype(primal_aval.dtype)
        assert expected_tangent_dtype == tangent_aval.dtype, (expected_tangent_dtype, tangent_aval.dtype)

call_param_updaters: Dict[core.Primitive, Callable] = {}
call_transpose_param_updaters: Dict[core.Primitive, Callable] = {}

# -------------------- Primitives --------------------

primitive_jvps: Dict[core.Primitive, Callable] = {}
primitive_transposes: Dict[core.Primitive, Callable] = {}
reducing_transposes: Dict[core.Primitive, Callable] = {}

def deflinear(primitive, transpose_rule):
    primitive_jvps[primitive] = partial(linear_jvp, primitive)
    primitive_transposes[primitive] = partial(linear_transpose, transpose_rule)

def linear_jvp(primitive, primals, tangents, **params):
    val_out = primitive.bind(*primals, **params)
    if all(isinstance(tangent, Zero) for tangent in tangents):
        if primitive.multiple_results:
            return val_out, [Zero.from_primal_value(v) for v in val_out]
        return val_out, Zero.from_primal_value(val_out)
    else:
        tangents = map(instantiate_zeros, tangents)
        return val_out, primitive.bind(*tangents, **params)

def linear_transpose(transpose_rule, cotangent, *args, **kwargs):
    return Zero if isinstance(cotangent, Zero) else transpose_rule(cotangent, **kwargs)

def deflinear2(primitive, transpose_rule):
    primitive_jvps[primitive] = partial(linear_jvp, primitive)
    primitive_transposes[primitive] = partial(linear_transpose2, transpose_rule)

def linear_transpose2(transpose_rule, cotangent, *args, **kwargs):
    return Zero if isinstance(cotangent, Zero) else transpose_rule(cotangent, *args, **kwargs)

def defjvp(primitive, *jvprules):
    assert isinstance(primitive, Primitive)
    assert not primitive.multiple_results
    primitive_jvps[primitive] = partial(standard_jvp, jvprules, primitive)

def standard_jvp(jvprules, primitive, primals, tangents, **params):
    val_out = primitive.bind(*primals, **params)
    tangents_out = [rule(t, *primals, **params) for rule, t in zip(jvprules, tangents)
                    if rule is not None and not isinstance(t, Zero)]
    return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_primal_value(val_out))

def defjvp2(primitive, *jvprules):
    assert isinstance(primitive, Primitive)
    assert not primitive.multiple_results
    primitive_jvps[primitive] = partial(standard_jvp2, jvprules, primitive)

def standard_jvp2(jvprules, primitive, primals, tangents, **params):
    val_out = primitive.bind(*primals, **params)
    tangents_out = (rule(t, val_out, *primals, **params) for rule, t in zip(jvprules, tangents)
                    if rule is not None and not isinstance(t, Zero))
    tangents_out = list(tangents_out)
    return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_primal_value(val_out))

def add_tangents(x, y):
    if isinstance(x, Zero):
        return y
    elif isinstance(y, Zero):
        return x
    else:
        return add_jaxvals(x, y)

def defbilinear(prim, lhs_rule, rhs_rule):
    assert isinstance(prim, Primitive)
    lhs_jvp = lambda g, x, y, **kwargs: prim.bind(g, y, **kwargs)
    rhs_jvp = lambda g, x, y, **kwargs: prim.bind(x, g, **kwargs)
    defjvp(prim, lhs_jvp, rhs_jvp)
    primitive_transposes[prim] = partial(bilinear_transpose, lhs_rule, rhs_rule)

def bilinear_transpose(lhs_rule, rhs_rule, cotangent, x, y, **kwargs):
    assert is_undefined_primal(x) ^ is_undefined_primal(y)
    if isinstance(cotangent, Zero):
        return Zero
    if is_undefined_primal(x):
        out = lhs_rule(cotangent, x, y, **kwargs)
        return Zero if out is Zero else (out, None)
    else:
        out = rhs_rule(cotangent, x, y, **kwargs)
        return Zero if out is Zero else (None, out)

def defjvp_zero(primitive):
    assert isinstance(primitive, Primitive)
    primitive_jvps[primitive] = partial(zero_jvp, primitive)

def zero_jvp(primitive, primals, tangents, **params):
    r = primitive.bind(*primals, **params)
    return r, Zero.from_primal_value(r)

deflinear2(add_jaxvals_p, lambda t, *args: (t, t))

def instantiate_zeros(tangent):
    return zeros_like_aval(tangent.aval) if isinstance(tangent, Zero) else tangent

@lu.transformation_with_aux
def traceable(in_tree, *primals_and_tangents):
    primals, tangents = tree_unflatten(in_tree, primals_and_tangents)
    tangents = [Zero.from_primal_value(p) if t is None else t
                for p, t in zip(primals, tangents)]
    primals_out, tangents_out = yield (primals, tangents), {}
    tangents_out = [None if isinstance(t, Zero) else t for t in tangents_out]
    out_flat, out_tree = tree_flatten((primals_out, tangents_out))
    yield out_flat, out_tree

def call_transpose(primitive, params, call_jaxpr, args, ct, _):
    if isinstance(call_jaxpr, core.ClosedJaxpr):
        call_jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
    else:
        consts = ()
    all_args, in_tree_def = tree_flatten((consts, args, ct))
    fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr, False)
    fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
    update_params = call_transpose_param_updaters.get(primitive)
    if update_params:
        params = update_params(params, map(is_undefined_primal, args),
                               [not isinstance(x, Zero) for x in ct])
    if config.dynamic_shapes.value:
        which_lin = [is_undefined_primal(x) for x in args]
        res_invars, _ = partition_list(which_lin, call_jaxpr.invars)
        new_invars = [*res_invars, *call_jaxpr.outvars]
        dbidx_map = {v: core.DBIdx(i) for i, v in enumerate(new_invars)}
        in_type = [(v.aval.update(shape=tuple(dbidx_map.get(d, d) for d in v.aval.shape))
                    if isinstance(v.aval, core.DShapedArray) else v.aval, True) for v in new_invars]
        fun = lu.annotate(fun, tuple(in_type))
    out_flat = primitive.bind(fun, *all_args, **params)
    return tree_unflatten(out_tree(), out_flat)
primitive_transposes[core.call_p] = partial(call_transpose, call_p)

def _closed_call_transpose(params, jaxpr, args, ct, cts_in_avals):
    jaxpr_, consts = jaxpr.jaxpr, jaxpr.consts
    jaxpr_ = pe.convert_constvars_jaxpr(jaxpr_)
    return call_transpose(core.closed_call_p, params, jaxpr_, (*consts, *args),
                          ct, cts_in_avals)
primitive_transposes[core.closed_call_p] = _closed_call_transpose

@lu.transformation_with_aux
def nonzero_outputs(*args, **kwargs):
    results = yield args, kwargs
    yield results, [not isinstance(r, Zero) for r in results]

def map_transpose(primitive, params, call_jaxpr, args, ct, _):
    all_args, in_tree_def = tree_flatten(((), args, ct))
    fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr, False)
    fun, nz_arg_cts = nonzero_outputs(fun)
    fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
    in_axes, out_axes = params['in_axes'], params['out_axes']
    new_in_axes = (*[axis for axis, x in zip(in_axes, args)
                     if not is_undefined_primal(x)],
                   *[axis for axis, x in zip(out_axes, ct)
                     if not isinstance(x, Zero)])
    if any(out_axis is None for out_axis in out_axes):
        raise NotImplementedError(
            "Autodiff of pmap functions with out_axes=None is not supported.")
    assert all(out_axis is not None for out_axis in out_axes), out_axes
    @as_hashable_function(closure=(in_axes, tuple(isinstance(c, Zero) for c in ct)))
    def out_axes_thunk():
        return tuple(axis or 0 for axis, nz in zip(in_axes, nz_arg_cts()) if nz)
    new_params = dict(params, name=wrap_name(params['name'], 'transpose'),
                      in_axes=new_in_axes, out_axes_thunk=out_axes_thunk)
    del new_params['out_axes']
    update_params = call_transpose_param_updaters.get(primitive)
    if update_params:
        new_params = update_params(new_params, map(is_undefined_primal, args),
                                   [not isinstance(x, Zero) for x in ct])
    out_flat = primitive.bind(fun, *all_args, **new_params)
    arg_cts = tree_unflatten(out_tree(), out_flat)

    assert len(in_axes) == len(arg_cts)
    def unmap_zero(zero, in_axis):
        return (zero if in_axis is None else
                Zero(core.unmapped_aval(params['axis_size'], params['axis_name'], in_axis, zero.aval)))
    arg_cts = (unmap_zero(arg_ct, in_axis) if isinstance(arg_ct, Zero) else
               arg_ct if in_axis is not None else
               arg_ct.sum(0)
               for arg_ct, in_axis in zip(arg_cts, in_axes))
    return tuple(arg_cts)

# Definition of custom_lin_p, which was missing previously
custom_lin_p: core.Primitive = core.Primitive('custom_lin')
custom_lin_p.def_abstract_eval(lambda *_, out_avals, **__: out_avals)
custom_lin_p.multiple_results = True

def raise_custom_vjp_error_on_jvp(*_, **__):
    raise TypeError("Can't apply forward-mode autodiff (jvp) to a custom_vjp function.")
custom_lin_p.def_impl(raise_custom_vjp_error_on_jvp)

def _custom_lin_transpose(cts_out, *invals, num_res, bwd, out_avals,
                          symbolic_zeros):
    res, _ = split_list(invals, [num_res])
    if symbolic_zeros:
        cts_out = map(replace_internal_symbolic_zeros, cts_out)
    else:
        cts_out = map(instantiate_zeros, cts_out)
    cts_in = bwd(*res, *cts_out)
    cts_in = map(replace_rule_output_symbolic_zeros, cts_in)
    return [None] * num_res + list(cts_in)
primitive_transposes[custom_lin_p] = _custom_lin_transpose

# Custom exception classes with improved error handling
class CustomJVPException(Exception):
    def __init__(self):
        msg = ("Detected differentiation of a custom_jvp function with respect to "
               "a closed-over value. That isn't supported because the custom JVP "
               "rule only specifies how to differentiate the custom_jvp function "
               "with respect to explicit input parameters. Try passing the "
               "closed-over value into the custom_jvp function as an argument, and "
               "adapting the custom_jvp rule.")
        super().__init__(msg)

class CustomVJPException(Exception):
    def __init__(self):
        msg = ("Detected differentiation of a custom_vjp function with respect to "
               "a closed-over value. That isn't supported because the custom VJP "
               "rule only specifies how to differentiate the custom_vjp function "
               "with respect to explicit input parameters. Try passing the "
               "closed-over value into the custom_vjp function as an argument, and "
               "adapting the custom_vjp fwd and bwd rules.")
        super().__init__(msg)
