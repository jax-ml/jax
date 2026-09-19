class GpuTargetConfig:
  platform_name: str
  device_description_str: str
  arch_name: str
  compute_capability: int
  core_count: int
  shared_memory_per_core: int

def get_gpu_spec(device_kind: str) -> GpuTargetConfig: ...
