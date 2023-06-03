# Copyright 2023 The JAX Authors.
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

import datetime
from numpy import array, float32


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_02 = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['tf.call_tf_function'],
    serialized_date=datetime.date(2023, 6, 2),
    inputs=(array([0.5, 0.7], dtype=float32),),
    expected_outputs=(array([0.88726   , 0.79956985], dtype=float32),),
    mlir_module_text=r"""
# First the MLIR module:
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<2xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_index = 0 : i64, has_token_input_output = false}} : (tensor<2xf32>) -> tensor<2xf32> loc(#loc2)
    %1 = stablehlo.cosine %0 : tensor<2xf32> loc(#loc3)
    return %1 : tensor<2xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":590:0)
#loc2 = loc("jit(func)/jit(main)/call_tf[callable_flat_tf=<function call_tf.<locals>.make_call.<locals>.callable_flat_tf at 0x7fafe24c55a0> function_flat_tf=<googlex.third_party.tensorflow.python.eager.polymorphic_function.polymorphic_function.Function object at 0x7fafe2510c40> args_flat_sig_tf=(TensorSpec(shape=(2,), dtype=tf.float32, name=None),) output_avals=(ShapedArray(float32[2]),) has_side_effects=True ordered=False call_tf_graph=True]"(#loc1))
#loc3 = loc("jit(func)/jit(main)/cos"(#loc1))

# Then the tf.Graph:
node {
  name: "the_input"
  op: "Placeholder"
  attr {
    key: "_user_specified_name"
    value {
      s: "the_input"
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
      }
    }
  }
}
node {
  name: "jax2tf_arg_0"
  op: "Identity"
  input: "the_input"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "XlaSharding"
  op: "XlaSharding"
  input: "jax2tf_arg_0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_XlaSharding"
    value {
      s: ""
    }
  }
  attr {
    key: "sharding"
    value {
      s: ""
    }
  }
  attr {
    key: "unspecified_dims"
    value {
      list {
      }
    }
  }
}
node {
  name: "XlaCallModule"
  op: "XlaCallModule"
  input: "XlaSharding"
  attr {
    key: "Sout"
    value {
      list {
        shape {
          dim {
            size: 2
          }
        }
      }
    }
  }
  attr {
    key: "Tin"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "Tout"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "dim_args_spec"
    value {
      list {
      }
    }
  }
  attr {
    key: "function_list"
    value {
      list {
        func {
          name: "__inference_callable_flat_tf_10"
        }
      }
    }
  }
  attr {
    key: "has_token_input_output"
    value {
      b: false
    }
  }
  attr {
    key: "module"
    value {
      s: "ML\357R\001StableHLO_v0.9.0\000\001\031\005\001\003\001\003\005\003\t\007\t\013\r\003\203e\013\0019\007\017\013\027#\013\013\0133\013\013\013\013S\013\013\013\013\013\013\013\013\013\017\013\013\017\013\003-\013\013\017\033\013\013\013\013\013\017\023\013\013\013\013\013\013\033\013\017\013\013\001\003\017\003\t\023\027\007\007\002\232\002\037\021\001\005\005\017\0273:\t\001\003\007\013\003\r\003\005\017\005\021\005\023\005\025\003\013\023=\025I\027K\005Q\031S\005\027\005\031\005\033\005\035\003\023\035U\037;!W#9%Y\'9)9+9-[\005\037\005!\005#\005%\005\'\005)\005+\005-\005/\0351\007\0051\0053\0357\007\0055\003\001\0357\003\003?\r\005ACEG\0359\035;\035=\035?#\005\003\003M\r\003O;\035A\035C\035E\013\005\035G\005\003\r\005]_ac\035I\023\t\001\035K\005\001\001\002\002)\003\t\007\021\003\003\003\003\t\035\004Q\005\001\021\001\t\007\003\001\005\003\021\001\021\005\003\007\017\003\003\001\005\007/\033\003\003\003\001\007\0065\003\003\003\003\t\004\001\003\005\006\003\001\005\001\000\312\017M/\033)\017\013!\033\035\005\033\0031\203\312\006%\037/!!)#\037\031\037\025\035\025\023%)\023\025\025\037\021\017\013\021builtin\000vhlo\000module\000func_v1\000custom_call_v1\000cosine_v1\000return_v1\000sym_name\000mhlo.num_partitions\000mhlo.num_replicas\000jit_func\000arg_attrs\000function_type\000res_attrs\000sym_visibility\000api_version\000backend_config\000call_target_name\000called_computations\000has_side_effect\000operand_layouts\000output_operand_aliases\000result_layouts\000tf.backend_config\000jit(func)/jit(main)/call_tf[callable_flat_tf=<function call_tf.<locals>.make_call.<locals>.callable_flat_tf at 0x7fafe24c55a0> function_flat_tf=<googlex.third_party.tensorflow.python.eager.polymorphic_function.polymorphic_function.Function object at 0x7fafe2510c40> args_flat_sig_tf=(TensorSpec(shape=(2,), dtype=tf.float32, name=None),) output_avals=(ShapedArray(float32[2]),) has_side_effects=True ordered=False call_tf_graph=True]\000third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\000jit(func)/jit(main)/cos\000\000jax.arg_info\000x\000mhlo.sharding\000{replicated}\000jax.result_info\000main\000public\000tf.call_tf_function\000called_index\000has_token_input_output\000"
    }
  }
  attr {
    key: "platforms"
    value {
      list {
        s: "CPU"
      }
    }
  }
  attr {
    key: "version"
    value {
      i: 5
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "XlaCallModule"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "IdentityN"
  op: "IdentityN"
  input: "XlaCallModule"
  input: "jax2tf_arg_0"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "_gradient_op_type"
    value {
      s: "CustomGradient-11"
    }
  }
}
node {
  name: "jax2tf_out"
  op: "Identity"
  input: "IdentityN"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "the_result"
  op: "Identity"
  input: "jax2tf_out"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Identity_1"
  op: "Identity"
  input: "the_result"
  input: "^NoOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "NoOp"
  op: "NoOp"
  input: "^XlaCallModule"
}
library {
  function {
    signature {
      name: "__inference_callable_flat_tf_10"
      input_arg {
        name: "args_tf_flat_0"
        type: DT_FLOAT
      }
      output_arg {
        name: "identity"
        type: DT_FLOAT
      }
    }
    node_def {
      name: "Sin"
      op: "Sin"
      input: "args_tf_flat_0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "EnsureShape"
      op: "EnsureShape"
      input: "Sin:y:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "shape"
        value {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    node_def {
      name: "Identity"
      op: "Identity"
      input: "EnsureShape:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    ret {
      key: "identity"
      value: "Identity:output:0"
    }
    attr {
      key: "_XlaMustCompile"
      value {
        b: false
      }
    }
    attr {
      key: "_construction_context"
      value {
        s: "kEagerRuntime"
      }
    }
    arg_attr {
      key: 0
      value {
        attr {
          key: "_output_shapes"
          value {
            list {
              shape {
                dim {
                  size: 2
                }
              }
            }
          }
        }
        attr {
          key: "_user_specified_name"
          value {
            s: "args_tf_flat_0"
          }
        }
      }
    }
  }
}
versions {
  producer: 1515
  min_consumer: 12
}
""",
    mlir_module_serialized=b'\n[\n\tthe_input\x12\x0bPlaceholder*\x0b\n\x05dtype\x12\x020\x01*\x0f\n\x05shape\x12\x06:\x04\x12\x02\x08\x02*#\n\x14_user_specified_name\x12\x0b\x12\tthe_input\n,\n\x0cjax2tf_arg_0\x12\x08Identity\x1a\tthe_input*\x07\n\x01T\x12\x020\x01\nm\n\x0bXlaSharding\x12\x0bXlaSharding\x1a\x0cjax2tf_arg_0*\x12\n\x0c_XlaSharding\x12\x02\x12\x00*\x16\n\x10unspecified_dims\x12\x02\n\x00*\x07\n\x01T\x12\x020\x01*\x0e\n\x08sharding\x12\x02\x12\x00\n\xaf\x0c\n\rXlaCallModule\x12\rXlaCallModule\x1a\x0bXlaSharding*6\n\rfunction_list\x12%\n#J!\n\x1f__inference_callable_flat_tf_10*\r\n\x07version\x12\x02\x18\x05*\x0c\n\x03Tin\x12\x05\n\x032\x01\x01*\x13\n\rdim_args_spec\x12\x02\n\x00*\x10\n\x04Sout\x12\x08\n\x06:\x04\x12\x02\x08\x02*\x1c\n\x16has_token_input_output\x12\x02(\x00*\xc2\n\n\x06module\x12\xb7\n\x12\xb4\nML\xefR\x01StableHLO_v0.9.0\x00\x01\x19\x05\x01\x03\x01\x03\x05\x03\t\x07\t\x0b\r\x03\x83e\x0b\x019\x07\x0f\x0b\x17#\x0b\x0b\x0b3\x0b\x0b\x0b\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03-\x0b\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0b\x01\x03\x0f\x03\t\x13\x17\x07\x07\x02\x9a\x02\x1f\x11\x01\x05\x05\x0f\x173:\t\x01\x03\x07\x0b\x03\r\x03\x05\x0f\x05\x11\x05\x13\x05\x15\x03\x0b\x13=\x15I\x17K\x05Q\x19S\x05\x17\x05\x19\x05\x1b\x05\x1d\x03\x13\x1dU\x1f;!W#9%Y\'9)9+9-[\x05\x1f\x05!\x05#\x05%\x05\'\x05)\x05+\x05-\x05/\x1d1\x07\x051\x053\x1d7\x07\x055\x03\x01\x1d7\x03\x03?\r\x05ACEG\x1d9\x1d;\x1d=\x1d?#\x05\x03\x03M\r\x03O;\x1dA\x1dC\x1dE\x0b\x05\x1dG\x05\x03\r\x05]_ac\x1dI\x13\t\x01\x1dK\x05\x01\x01\x02\x02)\x03\t\x07\x11\x03\x03\x03\x03\t\x1d\x04Q\x05\x01\x11\x01\t\x07\x03\x01\x05\x03\x11\x01\x11\x05\x03\x07\x0f\x03\x03\x01\x05\x07/\x1b\x03\x03\x03\x01\x07\x065\x03\x03\x03\x03\t\x04\x01\x03\x05\x06\x03\x01\x05\x01\x00\xca\x0fM/\x1b)\x0f\x0b!\x1b\x1d\x05\x1b\x031\x83\xca\x06%\x1f/!!)#\x1f\x19\x1f\x15\x1d\x15\x13%)\x13\x15\x15\x1f\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00custom_call_v1\x00cosine_v1\x00return_v1\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00tf.backend_config\x00jit(func)/jit(main)/call_tf[callable_flat_tf=<function call_tf.<locals>.make_call.<locals>.callable_flat_tf at 0x7fafe24c55a0> function_flat_tf=<googlex.third_party.tensorflow.python.eager.polymorphic_function.polymorphic_function.Function object at 0x7fafe2510c40> args_flat_sig_tf=(TensorSpec(shape=(2,), dtype=tf.float32, name=None),) output_avals=(ShapedArray(float32[2]),) has_side_effects=True ordered=False call_tf_graph=True]\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(func)/jit(main)/cos\x00\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00tf.call_tf_function\x00called_index\x00has_token_input_output\x00*\x14\n\tplatforms\x12\x07\n\x05\x12\x03CPU*\r\n\x04Tout\x12\x05\n\x032\x01\x01\n,\n\x08Identity\x12\x08Identity\x1a\rXlaCallModule*\x07\n\x01T\x12\x020\x01\nj\n\tIdentityN\x12\tIdentityN\x1a\rXlaCallModule\x1a\x0cjax2tf_arg_0*(\n\x11_gradient_op_type\x12\x13\x12\x11CustomGradient-11*\x0b\n\x01T\x12\x06\n\x042\x02\x01\x01\n*\n\njax2tf_out\x12\x08Identity\x1a\tIdentityN*\x07\n\x01T\x12\x020\x01\n+\n\nthe_result\x12\x08Identity\x1a\njax2tf_out*\x07\n\x01T\x12\x020\x01\n2\n\nIdentity_1\x12\x08Identity\x1a\nthe_result\x1a\x05^NoOp*\x07\n\x01T\x12\x020\x01\n\x1c\n\x04NoOp\x12\x04NoOp\x1a\x0e^XlaCallModule\x12\x8d\x03\n\x8a\x03:J\x08\x00\x12F\n(\n\x14_user_specified_name\x12\x10\x12\x0eargs_tf_flat_0\n\x1a\n\x0e_output_shapes\x12\x08\n\x06:\x04\x12\x02\x08\x02*(\n\x15_construction_context\x12\x0f\x12\rkEagerRuntime*\x15\n\x0f_XlaMustCompile\x12\x02(\x00"\x1d\n\x08identity\x12\x11Identity:output:0\x1a#\n\x03Sin\x12\x03Sin\x1a\x0eargs_tf_flat_0*\x07\n\x01T\x12\x020\x01\x1a=\n\x0bEnsureShape\x12\x0bEnsureShape\x1a\x07Sin:y:0*\x07\n\x01T\x12\x020\x01*\x0f\n\x05shape\x12\x06:\x04\x12\x02\x08\x02\x1a3\n\x08Identity\x12\x08Identity\x1a\x14EnsureShape:output:0*\x07\n\x01T\x12\x020\x01\nC\n\x1f__inference_callable_flat_tf_10\x12\x12\n\x0eargs_tf_flat_0\x18\x01\x1a\x0c\n\x08identity\x18\x01"\x05\x08\xeb\x0b\x10\x0c',
    xla_call_module_version=5,
)  # End paste
