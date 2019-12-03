from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from jax.lib import xla_bridge as xb


class XlaBridgeTest(absltest.TestCase):

  def test_set_device_assignment_no_partition(self):
    compile_options = xb.get_compile_options(
        num_replicas=4, device_assignment=[0, 1, 2, 3])
    expected_device_assignment = ("Computations: 1 Replicas: 4\nComputation 0: "
                                  "0 1 2 3 \n")
    self.assertEqual(compile_options.device_assignment.__repr__(),
                     expected_device_assignment)

  def test_set_device_assignment_with_partition(self):
    compile_options = xb.get_compile_options(
        num_replicas=2, device_assignment=[[0, 1], [2, 3]])
    expected_device_assignment = ("Computations: 2 Replicas: 2\nComputation 0: "
                                  "0 2 \nComputation 1: 1 3 \n")
    self.assertEqual(compile_options.device_assignment.__repr__(),
                     expected_device_assignment)


if __name__ == "__main__":
  absltest.main()
