.PHONY: docker clean

docker: .docker-jax-ubu22 .docker-jax-ubu24


.docker-jax-ubu24:
	docker build -f build/rocm/docker/Dockerfile.jax-rocm-ubu24 --tag jax-ubu24 --progress plain \
		--build-arg=ROCM_VERSION=6.2 \
		--build-arg=JAX_VERSION=$(shell python setup.py -V) \
		--build-arg=JAX_COMMIT=$(shell git rev-parse HEAD) \
		.
	touch .docker-jax-ubu24


clean-docker-ubu24:
	docker rmi jax-ubu24
	rm .docker-jax-ubu24


.docker-jax-ubu22:
	docker build -f build/rocm/docker/Dockerfile.jax-rocm-ubu22 --tag jax-ubu22 --progress plain \
		--build-arg=ROCM_VERSION=6.1.2 \
		--build-arg=JAX_VERSION=$(shell python setup.py -V) \
		--build-arg=JAX_COMMIT=$(shell git rev-parse HEAD) \
		.
	touch .docker-jax-ubu22


clean-docker-ubu22:
	docker rmi jax-ubu22
	rm .docker-jax-ubu22


clean: clean-docker-ubu22 clean-docker-ubu24
