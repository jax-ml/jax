#!/bin/bash
set -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
  exit 1
fi

export CC=/dt7/usr/bin/gcc
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

if [[ $# != 2 ]]
then
  usage
fi

# Builds and activates a specific Python version.
pyenv local "$PY_VERSION"

PY_TAG=$(python -c "import wheel; import wheel.pep425tags as t; print(t.get_abbr_impl() + t.get_impl_ver())")

echo "Python tag: $PY_TAG"

case $2 in
  cuda-included)
    python build.py --enable_cuda --bazel_startup_options="--output_user_root=/build/root"
    python include_cuda.py
    PLAT_NAME="manylinux2010_x86_64"
    ;;
  cuda)
    python build.py --enable_cuda --bazel_startup_options="--output_user_root=/build/root"
    PLAT_NAME="linux_x86_64"
    ;;
  nocuda)
    python build.py --enable_cuda --include_gpu_backend_if_cuda_enabled=false \
      --bazel_startup_options="--output_user_root=/build/root"
    PLAT_NAME="manylinux2010_x86_64"
    ;;
  *)
    usage
esac

python setup.py bdist_wheel --python-tag "$PY_TAG" --plat-name "$PLAT_NAME"
cp -r dist/* /dist
