from setuptools import setup

setup(
    name='jax',
    version='0.0',
    description='Differentiate, compile, and transform Numpy code.',
    author='JAX team',
    author_email='jax-team@google.com',
    packages=['jax', 'jax.lib', 'jax.interpreters', 'jax.numpy', 'jax.scipy',
              'jax.experimental'],
    install_requires=['numpy>=1.12', 'six', 'protobuf'],
    url='https://github.com/google/jax',
    license='Apache-2.0',
)
