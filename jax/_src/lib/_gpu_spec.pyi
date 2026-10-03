class GpuTargetConfig:
  platform_name: str
  device_description_str: str
  arch_name: str
  compute_capability: int
  core_count: int
  smem_capacity_bytes: int

class GpuModel: ...

def get_gpu_spec(gpu_model: GpuModel) -> GpuTargetConfig: ...
