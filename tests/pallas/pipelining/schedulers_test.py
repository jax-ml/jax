# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.pallas.pipelining import internal
from jax._src.pallas.pipelining import pipeline_test_util as test_util
from jax._src.pallas.pipelining import schedulers


jax.config.parse_flags_with_absl()


def empty_jaxpr():
  def noop():
    pass
  return jax.make_jaxpr(noop)


class SchedulersGoldenTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("Only works on CPU")

  def test_2_async_stages(self):
    # This test uses 2 stages that are both async.
    # 1
    # |
    # 2
    token1 = internal.make_token("a")
    token2 = internal.make_token("b")
    stage1_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token1),),
        properties=internal.SchedulingProperties(
            max_in_flight=3, is_async_start=True, is_async_done=False),
        name="stage1_start"
    )
    stage1_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1), internal.WriteEffect(0)),
        properties=internal.SchedulingProperties(
            max_in_flight=3, is_async_start=False, is_async_done=True),
        name="stage1_end"
    )
    stage2_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token2),
                 # We need to insert this token so that stage1_start
                 # does not clobber input 0.
                 internal.ReadEffect(token1),
                 internal.ReadEffect(0)),
        properties=internal.SchedulingProperties(
            max_in_flight=3, is_async_start=True, is_async_done=False),
        name="stage2_start"
    )
    stage2_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token2),),
        properties=internal.SchedulingProperties(
            max_in_flight=3, is_async_start=False, is_async_done=True),
        name="stage2_end"
    )
    loop_struct = internal.NDLoopStruct(
        stages=(stage1_start, stage1_done, stage2_start, stage2_done),
        grid=(4,)
    )
    with jtu.capture_stdout() as stdout:
      schedulers.static_nd_loop_scheduler(
          loop_struct,
          args=(None,),
          eval_fn=test_util.print_stage)
    output = stdout().strip().split("\n")
    expected = [
        "[itr=0] stage1_start",
        "[itr=1] stage1_start",
        "[itr=2] stage1_start",
        "[itr=0] stage1_end",
        "[itr=0] stage2_start",
        "[itr=3] stage1_start",
        "[itr=1] stage1_end",
        "[itr=1] stage2_start",
        "[itr=2] stage1_end",
        "[itr=2] stage2_start",
        "[itr=3] stage1_end",
        "[itr=0] stage2_end",
        "[itr=3] stage2_start",
        "[itr=1] stage2_end",
        "[itr=2] stage2_end",
        "[itr=3] stage2_end",
    ]
    self.assertEqual(output, expected)

  def test_async_inputs_with_different_buffering(self):
    # This test uses 2 input stages (1a, 1b) that feed into a synchronous stage.
    # 1a   1b
    #  \   /
    #    2
    token1a = internal.make_token("1a")
    token1b = internal.make_token("1b")
    stage1a_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token1a),),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=True, is_async_done=False),
        name="stage1a_start"
    )
    stage1a_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1a), internal.WriteEffect(0)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=True),
        name="stage1a_end"
    )
    stage1b_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token1b),),
        properties=internal.SchedulingProperties(
            max_in_flight=4, is_async_start=True, is_async_done=False),
        name="stage1b_start"
    )
    stage1b_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1b), internal.WriteEffect(1)),
        properties=internal.SchedulingProperties(
            max_in_flight=4, is_async_start=False, is_async_done=True),
        name="stage1b_end"
    )
    stage2 = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1a),
                 internal.ReadEffect(token1b),
                 internal.ReadEffect(0),
                 internal.ReadEffect(1)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=False),
        name="stage2"
    )
    loop_struct = internal.NDLoopStruct(
        stages=(stage1a_start, stage1a_done,
                stage1b_start, stage1b_done,
                stage2,),
        grid=(6,)
    )
    with jtu.capture_stdout() as stdout:
      schedulers.static_nd_loop_scheduler(
          loop_struct,
          args=(None, None),
          eval_fn=test_util.print_stage)
    output = stdout().strip().split("\n")
    expected = [
        "[itr=0] stage1a_start",
        "[itr=0] stage1b_start",
        "[itr=1] stage1b_start",
        "[itr=2] stage1b_start",
        "[itr=1] stage1a_start",
        "[itr=3] stage1b_start",
        "[itr=0] stage1a_end",
        "[itr=0] stage1b_end",
        "[itr=0] stage2",
        "[itr=2] stage1a_start",
        "[itr=4] stage1b_start",
        "[itr=1] stage1a_end",
        "[itr=1] stage1b_end",
        "[itr=1] stage2",
        "[itr=3] stage1a_start",
        "[itr=5] stage1b_start",
        "[itr=2] stage1a_end",
        "[itr=2] stage1b_end",
        "[itr=2] stage2",
        "[itr=4] stage1a_start",
        "[itr=3] stage1b_end",
        "[itr=3] stage1a_end",
        "[itr=3] stage2",
        "[itr=5] stage1a_start",
        "[itr=4] stage1b_end",
        "[itr=4] stage1a_end",
        "[itr=4] stage2",
        "[itr=5] stage1a_end",
        "[itr=5] stage1b_end",
        "[itr=5] stage2",
    ]
    self.assertEqual(output, expected)

  def test_synchronous_3_stage(self):
    # This test models a 3-stage pipeline where all stages are synchronous.
    # 1
    # |
    # 2
    # |
    # 3
    stage1 = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(0),),
        properties=internal.SchedulingProperties(
            max_in_flight=3, is_async_start=False, is_async_done=False),
        name="stage1"
    )
    stage2 = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(0), internal.WriteEffect(1),),
        properties=internal.SchedulingProperties(
            max_in_flight=3, is_async_start=False, is_async_done=False),
        name="stage2"
    )
    stage3 = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(1),),
        properties=internal.SchedulingProperties(
            max_in_flight=3, is_async_start=False, is_async_done=False),
        name="stage3"
    )
    loop_struct = internal.NDLoopStruct(
        stages=(stage1, stage2, stage3),
        grid=(4,)
    )

    with jtu.capture_stdout() as stdout:
      schedulers.static_nd_loop_scheduler(
          loop_struct,
          args=(None, None),
          eval_fn=test_util.print_stage)
    output = stdout().strip().split("\n")
    expected = [
        # step
        "[itr=0] stage1",
        # step
        "[itr=1] stage1",
        "[itr=0] stage2",
        # step
        "[itr=2] stage1",
        "[itr=1] stage2",
        "[itr=0] stage3",
        # step
        "[itr=3] stage1",
        "[itr=2] stage2",
        "[itr=1] stage3",
        # step
        "[itr=3] stage2",
        "[itr=2] stage3",
        # step
        "[itr=3] stage3",
    ]
    self.assertEqual(output, expected)

  def test_standard_emit_pipeline(self):
    # This test uses 3 stages where copy_in and copy_out are async.
    # copy_in
    #   |
    #  body
    #   |
    # copy_out
    token1 = internal.make_token("a")
    token2 = internal.make_token("b")
    copy_in_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token1),),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=True, is_async_done=False),
        name="copy_in_start"
    )
    copy_in_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1), internal.WriteEffect(0)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=True),
        name="copy_in_done"
    )
    body_stage = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1), internal.ReadEffect(0),
                 internal.WriteEffect(1)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=False),
        name="body"
    )
    copy_out_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token2),
                 internal.ReadEffect(1)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=True, is_async_done=False),
        name="copy_out_start"
    )
    copy_out_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token2),),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=True),
        name="copy_out_done"
    )
    loop_struct = internal.NDLoopStruct(
        stages=(copy_in_start, copy_in_done, body_stage,
                copy_out_start, copy_out_done),
        grid=(4, 4)
    )
    with jtu.capture_stdout() as stdout:
      schedulers.static_nd_loop_scheduler(
          loop_struct,
          args=(None,),
          eval_fn=test_util.print_stage)
    output = stdout().strip().split("\n")
    prologue = [
        "[itr=0] copy_in_start",
        "[itr=1] copy_in_start",
        "[itr=0] copy_in_done",
        "[itr=0] body",
        "[itr=0] copy_out_start",
        "[itr=2] copy_in_start",
        "[itr=1] copy_in_done",
        "[itr=1] body",
        "[itr=1] copy_out_start",
    ]
    steady_state = []
    for itr in range(3, 16):
      steady_state.extend([
          f"[itr={itr}] copy_in_start",
          test_util.AnyOrder([
              f"[itr={itr-3}] copy_out_done",
              f"[itr={itr-1}] copy_in_done",]),
          f"[itr={itr-1}] body",
          f"[itr={itr-1}] copy_out_start",
      ])
    epilogue = [
        "[itr=15] copy_in_done",
        "[itr=13] copy_out_done",
        "[itr=15] body",
        "[itr=15] copy_out_start",
        "[itr=14] copy_out_done",
        "[itr=15] copy_out_done",
    ]
    expected = prologue + steady_state + epilogue
    list_equal = test_util.compare_lists(output, expected)
    self.assertTrue(list_equal)

  def test_pipelined_prefetch(self):
    # This test uses 4 stages where prefetch, copy_in and copy_out are async.
    # prefetch
    #   |
    # copy_in
    #   |
    #  body
    #   |
    # copy_out
    token1 = internal.make_token("a")
    token2 = internal.make_token("b")
    token_prefetch = internal.make_token("c")
    prefetch_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token_prefetch),),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=True, is_async_done=False),
        name="prefetch_start"
    )
    prefetch_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token_prefetch), internal.WriteEffect(0)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=True),
        name="prefetch_done"
    )
    copy_in_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token_prefetch),
                 internal.ReadEffect(0),
                 internal.WriteEffect(token1),),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=True, is_async_done=False),
        name="copy_in_start"
    )
    copy_in_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1), internal.WriteEffect(1)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=True),
        name="copy_in_done"
    )
    body_stage = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token1), internal.ReadEffect(1),
                 internal.WriteEffect(2)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=False),
        name="body"
    )
    copy_out_start = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.WriteEffect(token2),
                 internal.ReadEffect(2)),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=True, is_async_done=False),
        name="copy_out_start"
    )
    copy_out_done = internal.PipelineStage(
        jaxpr=empty_jaxpr(),
        effects=(internal.ReadEffect(token2),),
        properties=internal.SchedulingProperties(
            max_in_flight=2, is_async_start=False, is_async_done=True),
        name="copy_out_done"
    )
    loop_struct = internal.NDLoopStruct(
        stages=(prefetch_start, prefetch_done,
                copy_in_start, copy_in_done,
                body_stage,
                copy_out_start, copy_out_done),
        grid=(4, 4)
    )
    with jtu.capture_stdout() as stdout:
      schedulers.static_nd_loop_scheduler(
          loop_struct,
          args=(None,),
          eval_fn=test_util.print_stage)
    output = stdout().strip().split("\n")
    # The schedule is slightly suboptimal here, noted in the comments.
    prologue = [
        "[itr=0] prefetch_start",
        "[itr=1] prefetch_start",
        "[itr=0] prefetch_done",
        "[itr=0] copy_in_start",
        "[itr=2] prefetch_start",
        "[itr=1] prefetch_done",
        "[itr=1] copy_in_start",
        "[itr=3] prefetch_start",
        "[itr=0] copy_in_done",
        # This can be pushed after body, before [itr=2] copy_in_start
        "[itr=2] prefetch_done",
        "[itr=0] body",
        "[itr=0] copy_out_start",
        "[itr=2] copy_in_start",
        "[itr=4] prefetch_start",
        "[itr=1] copy_in_done",
        # This can be pushed after body, before [itr=2] copy_in_start
        "[itr=3] prefetch_done",
        "[itr=1] body",
        "[itr=1] copy_out_start",
        "[itr=3] copy_in_start",
        # This can be pushed after [itr=5] prefetch_start
        "[itr=0] copy_out_done",
        "[itr=5] prefetch_start",
        "[itr=2] copy_in_done",
        "[itr=4] prefetch_done",
        "[itr=2] body",
        "[itr=2] copy_out_start",
    ]
    steady_state = []
    for i in range(6, 16):
      steady_state.extend([
          f"[itr={i-2}] copy_in_start",
          f"[itr={i-5}] copy_out_done",
          f"[itr={i}] prefetch_start",
          f"[itr={i-3}] copy_in_done",
          f"[itr={i-1}] prefetch_done",
          f"[itr={i-3}] body",
          f"[itr={i-3}] copy_out_start",
      ])
    epilogue = [
        "[itr=14] copy_in_start",
        "[itr=11] copy_out_done",
        "[itr=15] prefetch_done",
        "[itr=13] copy_in_done",
        "[itr=13] body",
        "[itr=13] copy_out_start",
        "[itr=15] copy_in_start",
        "[itr=12] copy_out_done",
        "[itr=14] copy_in_done",
        "[itr=14] body",
        "[itr=14] copy_out_start",
        "[itr=15] copy_in_done",
        "[itr=13] copy_out_done",
        "[itr=15] body",
        "[itr=15] copy_out_start",
        "[itr=14] copy_out_done",
        "[itr=15] copy_out_done",
    ]
    expected = prologue + steady_state + epilogue
    self.assertEqual(output, expected)

if __name__ == "__main__":
  absltest.main()
