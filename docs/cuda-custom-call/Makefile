NVCC = nvcc
NVCCFLAGS += -I$(shell python -c 'import jax; print(jax.ffi.include_dir())')

check: mul.so
	python mul.py

%.so: %.cu
	$(NVCC) $(NVCCFLAGS) --compiler-options=-shared,-fPIC -o $@ $<

clean:
	rm -rf *.so
