import numpy as _np
from jax.numpy import fft, linalg
from typing import Any, Callable, Dict, Tuple, Type, Union
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.index_tricks import _Mgrid, _Ogrid, CClass as _CClass, RClass as _RClass
from jax._src.numpy.reductions import CumulativeReduction as _CumulativeReduction
from jax._src.numpy.ufunc_api import ufunc as ufunc

ComplexWarning: Any
abs: Any
absolute: Any
add: Callable[[ArrayLike, ArrayLike], Array]
all: Any
allclose: Any
alltrue: Any
amax: Any
amin: Any
angle: Any
any: Any
append: Any
apply_along_axis: Any
apply_over_axes: Any
arange: Any
arccos: Callable[[ArrayLike], Array]
arccosh: Any
arcsin: Callable[[ArrayLike], Array]
arcsinh: Callable[[ArrayLike], Array]
arctan: Callable[[ArrayLike], Array]
arctan2: Callable[[ArrayLike, ArrayLike], Array]
arctanh: Callable[[ArrayLike], Array]
argmax: Any
argmin: Any
argpartition: Any
argsort: Any
argwhere: Any
around: Any
array: Any
array_equal: Any
array_equiv: Any
array_repr: Any
array_split: Any
array_str: Any
asarray: Any
atleast_1d: Any
atleast_2d: Any
atleast_3d: Any
average: Any
bartlett: Any
bfloat16: Any
bincount: Any
bitwise_and: Callable[[ArrayLike, ArrayLike], Array]
bitwise_not: Callable[[ArrayLike], Array]
bitwise_or: Callable[[ArrayLike, ArrayLike], Array]
bitwise_xor: Callable[[ArrayLike, ArrayLike], Array]
blackman: Any
block: Any
bool_: Any
broadcast_arrays: Any
broadcast_shapes: Any
broadcast_to: Any
c_: _CClass
can_cast: Any
cbrt: Callable[[ArrayLike], Array]
cdouble: Any
ceil: Callable[[ArrayLike], Array]
character: Any
choose: Any
clip: Any
column_stack: Any
complex128: Any
complex64: Any
complex_: Any
complexfloating: Any
compress: Any
concatenate: Any
conj: Any
conjugate: Any
convolve: Any
copy: Any
copysign: Any
corrcoef: Any
correlate: Any
cos: Callable[[ArrayLike], Array]
cosh: Callable[[ArrayLike], Array]
count_nonzero: Any
cov: Any
cross: Any
csingle: Any
cumprod: _CumulativeReduction
cumsum: _CumulativeReduction
deg2rad: Any
degrees: Any
delete: Any
diag: Any
diag_indices: Any
diag_indices_from: Any
diagflat: Any
diagonal: Any
diff: Any
digitize: Any
divide: Any
divmod: Any
dot: Any
double: Any
dsplit: Any
dstack: Any
dtype: Any
e: Any
ediff1d: Any
einsum: Any
einsum_path: Any
empty: Any
empty_like: Any
equal: Callable[[ArrayLike, ArrayLike], Array]
euler_gamma: Any
exp: Callable[[ArrayLike], Array]
exp2: Any
expand_dims: Any
expm1: Callable[[ArrayLike], Array]
extract: Any
eye: Any
fabs: Callable[[ArrayLike], Array]
finfo: Any
fix: Any
flatnonzero: Any
flexible: Any
flip: Any
fliplr: Any
flipud: Any
float16: Any
float32: Any
float64: Any
float8_e4m3b11fnuz: Any
float8_e4m3fn: Any
float8_e4m3fnuz: Any
float8_e5m2: Any
float8_e5m2fnuz: Any
float_: Any
float_power: Callable[[ArrayLike, ArrayLike], Array]
floating: Any
floor: Callable[[ArrayLike], Array]
floor_divide: Any
fmax: Any
fmin: Any
fmod: Any
frexp: Any
from_dlpack: Any
frombuffer: Any
fromfile: Any
fromfunction: Any
fromiter: Any
def frompyfunc(func, /, nin, nout, *, identity = ...) -> ufunc: ...
fromstring: Any
full: Any
full_like: Any
gcd: Any
generic: Any
geomspace: Any
get_printoptions: Any
gradient: Any
greater: Callable[[ArrayLike, ArrayLike], Array]
greater_equal: Callable[[ArrayLike, ArrayLike], Array]
hamming: Any
hanning: Any
heaviside: Any
histogram: Any
histogram2d: Any
histogram_bin_edges: Any
histogramdd: Any
hsplit: Any
hstack: Any
hypot: Any
i0: Any
identity: Any
iinfo: Any
imag: Any
in1d: Any
index_exp: Any
indices: Any
inexact: Any
inf: Any
inner: Any
insert: Any
int16: Any
int32: Any
int4: Any
int64: Any
int8: Any
int_: Any
integer: Any
interp: Any
intersect1d: Any
invert: Callable[[ArrayLike], Array]
isclose: Any
iscomplex: Any
iscomplexobj: Any
isfinite: Any
isin: Any
isinf: Any
isnan: Any
isneginf: Callable[[ArrayLike], Array]
isposinf: Callable[[ArrayLike], Array]
isreal: Any
isrealobj: Any
isscalar: Any
issubdtype: Any
issubsctype: Any
iterable: Any
ix_: Any
kaiser: Any
kron: Any
lcm: Any
ldexp: Any
left_shift: Callable[[ArrayLike, ArrayLike], Array]
less: Callable[[ArrayLike, ArrayLike], Array]
less_equal: Callable[[ArrayLike, ArrayLike], Array]
lexsort: Any
linspace: Any
load: Any
log: Callable[[ArrayLike], Array]
log10: Any
log1p: Callable[[ArrayLike], Array]
log2: Any
logaddexp: Any
logaddexp2: Any
logical_and: Callable[[ArrayLike, ArrayLike], Array]
logical_not: Callable[[ArrayLike], Array]
logical_or: Callable[[ArrayLike, ArrayLike], Array]
logical_xor: Callable[[ArrayLike, ArrayLike], Array]
logspace: Any
mask_indices: Any
matmul: Any
matrix_transpose: Any
max: Any
maximum: Callable[[ArrayLike, ArrayLike], Array]
mean: Any
median: Any
meshgrid: Any
mgrid: _Mgrid
min: Any
minimum: Callable[[ArrayLike, ArrayLike], Array]
mod: Any
modf: Any
moveaxis: Any
multiply: Callable[[ArrayLike, ArrayLike], Array]
nan: Any
nan_to_num: Any
nanargmax: Any
nanargmin: Any
nancumprod: _CumulativeReduction
nancumsum: _CumulativeReduction
nanmax: Any
nanmean: Any
nanmedian: Any
nanmin: Any
nanpercentile: Any
nanprod: Any
nanquantile: Any
nanstd: Any
nansum: Any
nanvar: Any
ndarray = Array
ndim: Any
negative: Callable[[ArrayLike], Array]
newaxis: Any
nextafter: Callable[[ArrayLike, ArrayLike], Array]
nonzero: Any
not_equal: Callable[[ArrayLike, ArrayLike], Array]
number: Any
object_: Any
ogrid: _Ogrid
ones: Any
ones_like: Any
outer: Any
packbits: Any
pad: Any
partition: Any
percentile: Any
pi: Any
piecewise: Any
place: Any
poly: Any
polyadd: Any
polyder: Any
polydiv: Any
polyfit: Any
polyint: Any
polymul: Any
polysub: Any
polyval: Any
positive: Callable[[ArrayLike], Array]
power: Any
printoptions: Any
prod: Any
product: Any
promote_types: Any
ptp: Any
put: Any
quantile: Any
r_: _RClass
rad2deg: Any
radians: Any
ravel: Any
ravel_multi_index: Any
real: Any
reciprocal: Any
register_jax_array_methods: Any
remainder: Any
repeat: Any
reshape: Any
resize: Any
result_type: Any
right_shift: Any
rint: Any
roll: Any
rollaxis: Any
roots: Any
rot90: Any
round: Any
round_: Any
row_stack: Any  # TODO(jakevdp): remove this
s_: Any
save: Any
savez: Any
searchsorted: Any
select: Any
set_printoptions: Any
setdiff1d: Any
setxor1d: Any
shape: Any
sign: Any
signbit: Any
signedinteger: Any
sin: Callable[[ArrayLike], Array]
sinc: Any
single: Any
sinh: Callable[[ArrayLike], Array]
size: Any
sort: Any
sort_complex: Any
split: Any
sqrt: Callable[[ArrayLike], Array]
square: Any
squeeze: Any
stack: Any
std: Any
subtract: Callable[[ArrayLike, ArrayLike], Array]
sum: Any
swapaxes: Any
take: Any
take_along_axis: Any
tan: Callable[[ArrayLike], Array]
tanh: Callable[[ArrayLike], Array]
tensordot: Any
tile: Any
trace: Any
transpose: Any
trapz: Any
tri: Any
tril: Any
tril_indices: Any
tril_indices_from: Any
trim_zeros: Any
triu: Any
triu_indices: Any
triu_indices_from: Any
true_divide: Any
trunc: Any
typing: Any
uint: Any
uint16: Any
uint32: Any
uint4: Any
uint64: Any
uint8: Any
union1d: Any
unique: Any
unpackbits: Any
unravel_index: Any
unsignedinteger: Any
unwrap: Any
vander: Any
var: Any
vdot: Any
def vectorize(pyfunc, *, excluded = ..., signature = ...) -> Callable: ...
vsplit: Any
vstack: Any
where: Any
zeros: Any
zeros_like: Any
