NVCC = nvcc
# Assumes a checkout of XLA is a sister directory. After
# https://github.com/google/jax/pull/21329 is merged, this should be replaced
# with an inline Python script to print the include directory, something along
# the lines of:
#   NVCCFLAGS += -I$(shell python -c 'import jax; print(jax.ffi.include_dir())')
NVCCFLAGS += -I../xla/
# Assumes pybind11 has been installed in the current environment, as simple as:
#   pip install pybind11
NVCCFLAGS += $(shell python -m pybind11 --includes)

check: mul.so
	python mul.py

%.so: %.cu
	$(NVCC) $(NVCCFLAGS) --compiler-options=-shared,-fPIC -o $@ $<

clean:
	rm -rf *.so
