# Primitives with limited support

| Affected primitive | Type of limitation | Description | Devices affected |
| --- | --- | --- | --- |
| atan2 | Missing TF support | Missing TF kernels for atan2 with dtype bfloat16 | CPU, GPU, TPU |
| atan2 | Missing TF support | Missing TF kernels for atan2 with dtype float16 | CPU, GPU, TPU |
| max | Missing TF support | max is unimplemented for dtype bool | CPU, GPU, TPU |
| max | Missing TF support | max is unimplemented for dtype complex64 | CPU, GPU, TPU |
| max | Missing TF support | max is unimplemented for dtype int8 | CPU, GPU, TPU |
| max | Missing TF support | max is unimplemented for dtype uint16 | CPU, GPU, TPU |
| max | Missing TF support | max is unimplemented for dtype uint32 | CPU, GPU, TPU |
| min | Missing TF support | min is unimplemented for dtype bool | CPU, GPU, TPU |
| min | Missing TF support | min is unimplemented for dtype complex64 | CPU, GPU, TPU |
| min | Missing TF support | min is unimplemented for dtype int8 | CPU, GPU, TPU |
| min | Missing TF support | min is unimplemented for dtype uint16 | CPU, GPU, TPU |
| min | Missing TF support | min is unimplemented for dtype uint32 | CPU, GPU, TPU |
| nextafter | Missing TF support | nextafter is unimplemented for dtype bfloat16 | CPU, GPU, TPU |
| nextafter | Missing TF support | nextafter is unimplemented for dtype float16 | CPU, GPU, TPU |
| rem | Missing TF support | Missing TF kernels for rem with dtype bfloat16 | CPU, GPU, TPU |
| rem | Missing TF support | Missing TF kernels for rem with dtype float16 | CPU, GPU, TPU |