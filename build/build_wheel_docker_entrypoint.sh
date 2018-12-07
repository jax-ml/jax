#!/bin/bash
set -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
  exit 1
fi

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

case $1 in
  py3)
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10
    ;;
  py2)
    ;;
  *)
    usage
esac

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

python setup.py bdist_wheel
cp -r dist/* /dist
