#!/bin/bash
set -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
  exit 1
fi

export CC=/dt7/usr/bin/gcc
export GCC_HOST_COMPILER_PATH=/dt7/usr/bin/gcc
export PYENV_ROOT="/pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

PY_VERSION="$1"
echo "Python version $PY_VERSION"

git clone https://github.com/google/jax /build/jax
cd /build/jax/build

mkdir /build/tmp
mkdir /build/root
export TMPDIR=/build/tmp

usage() {
  echo "usage: ${0##*/} [py2|py3] [cuda-included|cuda|nocuda]"
  exit 1
}

if [[ $# -lt 2 ]]
then
  usage
fi

# Builds and activates a specific Python version.
pyenv local "$PY_VERSION"

PY_TAG=$(python -c "import packaging.tags as t; print(t.interpreter_name() + t.interpreter_version())")

echo "Python tag: $PY_TAG"

# Workaround for https://github.com/bazelbuild/bazel/issues/9254
export BAZEL_LINKLIBS="-lstdc++"

case $2 in
  cuda-included)
    python build.py --enable_cuda --bazel_startup_options="--output_user_root=/build/root"
    python include_cuda.py
    PLAT_NAME="manylinux2010_x86_64"
    ;;
  cuda)
    python build.py --enable_cuda --bazel_startup_options="--output_user_root=/build/root"
    PLAT_NAME="manylinux2010_x86_64"
    ;;
  nocuda)
    python build.py --bazel_startup_options="--output_user_root=/build/root"
    PLAT_NAME="manylinux2010_x86_64"
    ;;
  *)
    usage
esac

export JAX_CUDA_VERSION=$3
python setup.py bdist_wheel --python-tag "$PY_TAG" --plat-name "$PLAT_NAME"
if python -m auditwheel show dist/jaxlib-*.whl  | grep -v 'platform tag: "manylinux2010_x86_64"' > /dev/null; then
  echo "jaxlib wheel is not manylinux2010 compliant"
  exit 1
fi
cp -r dist/* /dist
