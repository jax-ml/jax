.PHONY: all clean

all: .docker-jax-ubu22 .docker-jax-ubu24

clean: clean-jax-ubu22 clean-jax-ubu24

ROCM_VERSION = 6.2.0

.docker-%  : build/rocm/docker/Dockerfile.%
	docker build -f $< --tag $(*F) --progress plain \
		--build-arg=ROCM_VERSION=${ROCM_VERSION} \
		--build-arg=JAX_VERSION=$(shell python setup.py -V) \
		--build-arg=JAX_COMMIT=$(shell git rev-parse HEAD) \
		.
	@touch $@


clean-%:
	-docker rmi $(*F)
	@rm -f .docker-$(*F)
