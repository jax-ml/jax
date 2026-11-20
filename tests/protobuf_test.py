# Copyright 2026 The JAX Authors.
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

import glob
import gzip
import os
import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src import protobuf as jax_protobuf_lib
from jax.experimental import protobuf as jpb
import jax.extend.xla as jex_xla
import jax.numpy as jnp

config.parse_flags_with_absl()


class ProtobufTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    try:
      jax_protobuf_lib.bundled_descriptor_pool()
    except (ImportError, RuntimeError) as e:
      self.skipTest(str(e))

  def _dump_hlo(self, dump_dir, **options):
    def my_function(x):
      return jnp.sin(x) * 2

    jax.jit(
        my_function,
        compiler_options={
            "xla_dump_to": dump_dir,
            "xla_dump_hlo_as_proto": True,
            **options,
        },
    )(jnp.arange(4.0))
    pattern = "*my_function*after_optimizations.hlo.pb"
    if options.get("xla_dump_compress_protos"):
      pattern += ".gz"
    files = glob.glob(os.path.join(dump_dir, pattern))
    self.assertLen(files, 1)
    return files[0]

  def _check_hlo(self, hlo):
    self.assertIn("my_function", hlo.hlo_module.name)
    opcodes = {
        instr.opcode
        for comp in hlo.hlo_module.computations
        for instr in comp.instructions
    }
    self.assertIn("parameter", opcodes)
    self.assertTrue(hlo.HasField("buffer_assignment"))

  def test_can_parse_hlo_dump_from_path(self):
    with tempfile.TemporaryDirectory() as dump_dir:
      hlo = jpb.parse_file(pathlib.Path(self._dump_hlo(dump_dir)),
                           "xla.HloProto")
    self.assertIsInstance(hlo, jax_protobuf_lib.message_class("xla.HloProto"))
    self._check_hlo(hlo)

  def test_parse_file_decompresses_gz_files(self):
    """parse_file reads .gz dumps, whose raw bytes parse() rejects."""
    from google.protobuf.message import DecodeError

    with tempfile.TemporaryDirectory() as dump_dir:
      path = self._dump_hlo(dump_dir, xla_dump_compress_protos=True)
      self._check_hlo(jpb.parse_file(path, "xla.HloProto"))
      # Compressed data is rejected rather than misparsed.
      with open(path, "rb") as f, self.assertRaises(DecodeError):
        jpb.parse(f.read(), "xla.HloProto")

  def test_can_parse_text_format_protos(self):
    """Text protos parse from str, and from .txt and .txt.gz files."""
    from google.protobuf import text_format

    with tempfile.TemporaryDirectory() as dump_dir:
      hlo = jpb.parse_file(self._dump_hlo(dump_dir), "xla.HloProto")
      text = text_format.MessageToString(hlo)
      self.assertEqual(jpb.parse(text, "xla.HloProto"), hlo)
      for name, opener in [("hlo.txt", open), ("hlo.txt.gz", gzip.open)]:
        path = os.path.join(dump_dir, name)
        with opener(path, "wt") as f:
          f.write(text)
        # Text format is inferred from the .txt suffix.
        self.assertEqual(jpb.parse_file(path, "xla.HloProto"), hlo)

  def test_parse_file_needs_text_option_for_unknown_suffix(self):
    """Only .pb and .txt suffixes are inferred; text= overrides the suffix."""
    msg = jax_protobuf_lib.message_class("xla.DebugOptions")(xla_dump_to="/x")
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "debug_options.binpb")
      with open(path, "wb") as f:
        f.write(msg.SerializeToString())
      with self.assertRaisesRegex(ValueError, "pass text=False or text=True"):
        jpb.parse_file(path, "xla.DebugOptions")
      # An explicit format overrides the suffix.
      self.assertEqual(
          jpb.parse_file(path, "xla.DebugOptions", text=False), msg)

  def test_can_parse_profiler_xspace(self):
    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        jnp.ones(4).block_until_ready()
      files = glob.glob(os.path.join(trace_dir, "**/*.xplane.pb"),
                        recursive=True)
      self.assertLen(files, 1)
      xspace = jpb.parse_file(files[0], "tensorflow.profiler.XSpace")
    self.assertIn("/host:CPU", [plane.name for plane in xspace.planes])

  def test_can_parse_types_from_editions_files(self):
    """xla.DebugOptions uses an enum defined in an Editions-syntax file."""
    debug_options = jax_protobuf_lib.message_class("xla.DebugOptions")
    field = debug_options.DESCRIPTOR.fields_by_name[
        "xla_autotuner_preferred_backend"]
    self.assertEqual(field.enum_type.full_name, "xla.autotuner.Backend")
    msg = debug_options(xla_autotuner_preferred_backend=1)
    self.assertEqual(
        jpb.parse(msg.SerializeToString(), "xla.DebugOptions"), msg)

  def test_jax_message_types_are_available(self):
    for message_type in ["jax.PyTreeDefProto", "jax.PyHostCallbackProto",
                         "jax_triton.TritonAnyKernelCall",
                         "mosaic.gpu.MosaicGpuKernelProto"]:
      with self.subTest(message_type):
        self.assertEqual(jpb.parse(b"", message_type).ByteSize(), 0)

  def test_can_unpack_jax_proto_from_any_field(self):
    """A JAX proto packed in an Any is found via the parsed message's pool."""
    from google.protobuf import message_factory
    from google.protobuf import text_format

    callback = jax_protobuf_lib.message_class("jax.PyHostCallbackProto")(
        callable=b"pickled")
    metadata = jax_protobuf_lib.message_class("xla.HloPassMetadata")()
    metadata.custom_metadata.Pack(callback)
    msg = jpb.parse(metadata.SerializeToString(), "xla.HloPassMetadata")

    # The packed type can be found in the pool the parsed message came from.
    pool = msg.DESCRIPTOR.file.pool
    cls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName(msg.custom_metadata.TypeName()))
    unpacked = cls()
    self.assertTrue(msg.custom_metadata.Unpack(unpacked))
    self.assertEqual(unpacked.callable, b"pickled")
    self.assertIn('callable: "pickled"',
                  text_format.MessageToString(msg, descriptor_pool=pool))

  def test_bundled_descriptor_set_includes_xla_protos(self):
    from google.protobuf import descriptor_pb2

    fds = descriptor_pb2.FileDescriptorSet.FromString(
        jax_protobuf_lib.bundled_descriptor_set_bytes())
    self.assertIn("xla/service/hlo.proto", {f.name for f in fds.file})

  def test_parse_uses_custom_pool(self):
    """A pool from load_pool replaces the bundled one, for binary and text."""
    from google.protobuf import descriptor_pb2

    fdp = descriptor_pb2.FileDescriptorProto(
        name="custom.proto", package="xla", syntax="proto3")
    msg = fdp.message_type.add(name="HloProto")
    msg.field.add(name="extra", number=1000, json_name="extra",
                  type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
                  label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL)
    custom_pool = jpb.load_pool(
        descriptor_pb2.FileDescriptorSet(file=[fdp]).SerializeToString())
    data = b"\xc2\x3e\x02hi"  # Field 1000 (length-delimited) = "hi".

    # The default descriptor set does not know about field 1000...
    default = jpb.parse(data, "xla.HloProto")
    self.assertNotIn("extra", default.DESCRIPTOR.fields_by_name)
    # ... but the custom one does, and is used for both binary and text.
    custom = jpb.parse(data, "xla.HloProto", pool=custom_pool)
    self.assertEqual(custom.extra, "hi")
    self.assertEqual(
        jpb.parse('extra: "hi"', "xla.HloProto", pool=custom_pool),
        custom)
    # Only the types in the custom descriptor set are available.
    with self.assertRaisesRegex(ValueError, "Unknown XLA message type"):
      jpb.parse(b"", "xla.DebugOptions", pool=custom_pool)

  def test_each_pool_has_its_own_message_classes(self):
    """Messages from different pools have different types, even if equal."""
    with tempfile.TemporaryDirectory() as dump_dir:
      with open(self._dump_hlo(dump_dir), "rb") as f:
        data = f.read()
    explicit = jpb.parse(
        data, "xla.HloProto",
        pool=jpb.load_pool(jax_protobuf_lib.bundled_descriptor_set_bytes()))
    implicit = jpb.parse(data, "xla.HloProto")
    # Each pool creates its own message classes, so the messages are compared
    # by content.
    self.assertIsNot(type(explicit), type(implicit))
    self.assertEqual(str(explicit), str(implicit))

  @parameterized.parameters(
      jex_xla.PipelineStage.PRE_SCHEDULER,
      jex_xla.PipelineStage.POST_SCHEDULER,
  )
  @jtu.thread_unsafe_test()
  def test_can_rewrite_hlo_proto_in_module_transformation(self, stage):
    """A registered HLO transformation can edit and re-serialize the proto."""
    if jtu.TEST_WITH_PERSISTENT_COMPILATION_CACHE.value:
      self.skipTest("XLA transforms are not part of the persistent cache key")

    def sin_to_cos(serialized_hlo: bytes) -> bytes | None:
      # Edit the HloModuleProto directly, not via the HloModule bindings.
      module = jpb.parse(serialized_hlo, "xla.HloModuleProto")
      changed = False
      for computation in module.computations:
        for instruction in computation.instructions:
          if instruction.opcode == "sine":
            instruction.opcode = "cosine"
            changed = True
      return module.SerializeToString() if changed else None

    name = f"protobuf_sin_to_cos_{stage.name}"
    jex_xla.register_hlo_module_transformation(sin_to_cos, name=name,
                                               stage=stage)
    self.addCleanup(jex_xla.clear_hlo_module_transformation, name, stage)

    x = jnp.array([0.0, 1.0, 2.0])
    self.assertAllClose(jax.jit(jnp.sin)(x), jnp.cos(x), atol=1e-5)

  def test_parsing_unqualified_message_type_fails(self):
    with self.assertRaisesRegex(ValueError, "Unknown XLA message type"):
      jpb.parse(b"", "HloProto")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
