NVCC = nvcc
NVCCFLAGS += -I$(shell python -c 'from jax.extend import ffi; print(ffi.include_dir())')
NVCCFLAGS += -arch native
# since the file extension is .cu.cc, need to explicitly tell NVCC to treat is as .cu
NVCCFLAGS += -x cu

check: foo.so
	python cuda_custom_call_test.py

%.so: %.cu.cc
	$(NVCC) $(NVCCFLAGS) --compiler-options=-shared,-fPIC -o $@ $<

clean:
	rm -rf *.so
