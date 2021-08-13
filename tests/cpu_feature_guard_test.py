from jaxlib import cpu_feature_guard

print(dir(cpu_feature_guard))
print(cpu_feature_guard.__doc__)

from jaxlib import cusparse_kernels
print(dir(cusparse_kernels))