NVCC = nvcc
NVCCFLAGS += -I$(shell python -c 'import jax; print(jax.ffi.include_dir())')
NVCCFLAGS += -arch native

check: foo.so
	python foo.py

%.so: %.cu
	$(NVCC) $(NVCCFLAGS) --compiler-options=-shared,-fPIC -o $@ $<

clean:
	rm -rf *.so
