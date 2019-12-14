
from .common cimport bitgen_t, uint32_t
cimport numpy as np

cdef class BitGenerator():
    cdef readonly object _seed_seq
    cdef readonly object lock
    cdef bitgen_t _bitgen
    cdef readonly object _ctypes
    cdef readonly object _cffi
    cdef readonly object capsule


cdef class SeedSequence():
    cdef readonly object entropy
    cdef readonly tuple spawn_key
    cdef readonly uint32_t pool_size
    cdef readonly object pool
    cdef readonly uint32_t n_children_spawned

    cdef mix_entropy(self, np.ndarray[np.npy_uint32, ndim=1] mixer,
                     np.ndarray[np.npy_uint32, ndim=1] entropy_array)
    cdef get_assembled_entropy(self)

cdef class SeedlessSequence():
    pass
