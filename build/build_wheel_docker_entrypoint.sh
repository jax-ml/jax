#!/bin/bash -xev
if [ ! -d "/dist" ]
then
  echo "/dist must be mounted to produce output"
  exit 1
fi

git clone -b binary-distros https://github.com/google/jax /build/jax
cd /build/jax

usage() {
  echo "usage: ${0##*/} [python2|python3] [cuda-included|cuda|nocuda]"
  exit 1
}

if [[ $# != 2 ]]
then
  usage
fi

case $1 in
  py2)
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10
    ;;
  py3)
    ;;
  *)
    usage
esac

case $2 in
  cuda-included)
    python build.py --enable_cuda --cudnn_path /usr/lib/x86_64-linux-gnu/
    python build/include_cuda.py
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

python setup.py bdist bdist_wheel
cp -r dist/* /dist
