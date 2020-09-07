# Primitives with limited support

| Affected primitive | Type of limitation | Description | Devices affected |
| --- | --- | --- | --- |
| add | Missing TF support | add is unimplemented for dtype uint16 | CPU, GPU, TPU |
| add | Missing TF support | add is unimplemented for dtype uint32 | CPU, GPU, TPU |
| atan2 | Missing TF support | atan2 is unimplemented for dtype bfloat16 | CPU, GPU, TPU |
| atan2 | Missing TF support | atan2 is unimplemented for dtype float16 | CPU, GPU, TPU |
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
| mul | Missing TF support | mul is unimplemented for dtype uint32 | CPU, GPU, TPU |
| nextafter | Missing TF support | nextafter is unimplemented for dtype bfloat16 | CPU, GPU, TPU |
| nextafter | Missing TF support | nextafter is unimplemented for dtype float16 | CPU, GPU, TPU |
| qr | Missing TF support | qr is unimplemented for dtype complex64 | CPU, GPU, TPU |
| reduce_window_sum | Missing TF support | reduce_window_sum is unimplemented for dtype uint16 | CPU, GPU, TPU |
| reduce_window_sum | Missing TF support | reduce_window_sum is unimplemented for dtype uint32 | CPU, GPU, TPU |
| rem | Missing TF support | rem is unimplemented for dtype bfloat16 | CPU, GPU, TPU |
| rem | Missing TF support | rem is unimplemented for dtype float16 | CPU, GPU, TPU |
| select_and_gather_add | Missing TF support | select_and_gather_add is unimplemented for dtype float32 | TPU |
| svd | Missing TF support | svd is unimplemented for dtype complex64; this works on JAX because JAX uses a custom implementation | CPU, GPU |
