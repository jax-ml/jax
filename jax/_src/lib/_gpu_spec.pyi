import enum


class GpuTargetConfig:
  @property
  def platform_name(self) -> str: ...
  @property
  def device_description_str(self) -> str: ...
  @property
  def arch_name(self) -> str: ...
  @property
  def compute_capability(self) -> int: ...
  @property
  def core_count(self) -> int: ...
  @property
  def smem_capacity_bytes(self) -> int: ...


class GpuModel(enum.Enum):
  A100_PCIE_80 = 0
  A100_SXM_40 = 1
  A100_SXM_80 = 2
  A6000 = 3
  B200 = 4
  B200_MIG = 5
  B300 = 6
  BMG_G21 = 7
  H100_PCIE = 8
  H100_SXM = 9
  H100_SXM_MIG = 10
  H200 = 11
  MI200 = 12
  MI300 = 13
  MI350 = 14
  MI450 = 15
  P100 = 16
  PVC = 17
  V100 = 18
  GB200 = 19
  GB300 = 20
  RTX6000PRO = 21

def get_gpu_spec(gpu_model: GpuModel | enum.Enum) -> GpuTargetConfig: ...
