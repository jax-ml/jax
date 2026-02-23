from typing import Callable, Sequence

from mlir import ir


class TrivialTransferPlan:
  """TrivialTransferPlan conforms to the `TransferPlan` protocol in `fragmented_array.py`."""

  def tile_index_transforms(
      self,
  ) -> Sequence[Callable[[tuple[int, ...]], tuple[int, ...]]]: ...

  def select(self, idx: Sequence[ir.Value]) -> ir.Value: ...

  def select_if_group(
      self, group_idx: int, old: ir.Value, new: ir.Value
  ) -> ir.Value: ...


class StaggeredTransferPlan:
  """StaggeredTransferPlan conforms to the `TransferPlan` protocol in `fragmented_array.py`."""
  stagger: int
  dim: int
  size: int
  group_pred: ir.Value

  def tile_index_transforms(
      self,
  ) -> Sequence[Callable[[tuple[int, ...]], tuple[int, ...]]]: ...

  def select(self, idx: Sequence[ir.Value]) -> ir.Value: ...

  def select_if_group(
      self, group_idx: int, old: ir.Value, new: ir.Value
  ) -> ir.Value: ...
