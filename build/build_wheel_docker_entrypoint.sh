#!/bin/bash
set -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
  exit 1
fi

export PYENV_ROOT="/pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

PY_VERSION="$1"
echo "Python version $PY_VERSION"

git clone https://github.com/google/jax /build/jax
cd /build/jax/build

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
echo "Python tag $PY_TAG"

case $2 in
  cuda-included)
    python build.py --enable_cuda --cudnn_path /usr/lib/x86_64-linux-gnu/
    python include_cuda.py
    ;;
  cuda)
    python build.py --enable_cuda --cudnn_path /usr/lib/x86_64-linux-gnu/
    ;;
  nocuda)
    python build.py
    ;;
  *)
    usage
esac

python setup.py bdist_wheel --python-tag "$PY_TAG" --plat-name "linux_x86_64"
cp -r dist/* /dist
