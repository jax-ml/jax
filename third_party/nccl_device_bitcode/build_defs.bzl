# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build rule for NCCL device API bitcode (libnccl_device.bc).

Mirrors NCCL's own bindings/ir/Makefile pipeline using hermetic Bazel toolchains:
  clang -emit-llvm → opt (internalize) → llvm-dis (strip ftz) → llvm-as
"""

def _find_file(files, suffix):
    for f in files:
        if f.path.endswith(suffix):
            return f
    return None

def _find_path(files, marker):
    for f in files:
        idx = f.path.find(marker)
        if idx >= 0:
            return f.path[:idx + len(marker.rstrip("/"))]
    return None

def _raw_include_dirs(cc_target):
    """Gets raw include/ dirs from CcInfo headers (skips virtual includes)."""
    if CcInfo not in cc_target:
        return [], []
    headers = cc_target[CcInfo].compilation_context.headers.to_list()
    dirs = {}
    for f in headers:
        if "/_virtual_includes/" in f.path:
            continue
        idx = f.path.rfind("/include/")
        if idx >= 0:
            dirs[f.path[:idx + len("/include")]] = True
    return dirs.keys(), headers

def _nccl_device_bitcode_impl(ctx):
    # NCCL < 2.28 has no device API; the repo rule emits an empty filegroup.
    if not ctx.files.srcs:
        out = ctx.actions.declare_file("libnccl_device.bc")
        ctx.actions.write(out, "")
        return [DefaultInfo(files = depset([out]))]

    nccl_target = ctx.attr.nccl_hdrs[0]
    if CcInfo not in nccl_target:
        fail("nccl_hdrs must provide CcInfo")
    nccl_cc = nccl_target[CcInfo].compilation_context
    nccl_headers = nccl_cc.headers.to_list()

    nccl_device_h = _find_file(nccl_headers, "src/include/nccl_device.h")
    if not nccl_device_h:
        fail("nccl_device.h not found")

    wrapper_impl = _find_file(ctx.files.srcs, "nccl_device_wrapper__impl.h") or \
                   _find_file(ctx.files.srcs, "nccl_device_wrapper_impl.h")
    if not wrapper_impl:
        fail("nccl_device_wrapper__impl.h not found")

    cuda_nvcc_path = _find_path(ctx.files._cuda_nvcc, "/nvvm/")
    if not cuda_nvcc_path:
        fail("Could not find nvvm/ in cuda_nvcc")
    cuda_nvcc_path = cuda_nvcc_path[:cuda_nvcc_path.rfind("/nvvm")]

    resource_dir = _find_path(ctx.files._clang_headers, "/staging")
    if not resource_dir:
        fail("Could not find clang resource dir")

    cuda_include_dirs, cuda_header_files = _raw_include_dirs(ctx.attr._cuda_hdrs)

    nccl_inc_flags = " ".join(
        ["-I" + f.dirname for f in ctx.files.srcs] +
        ["-I" + d for d in nccl_cc.includes.to_list()] +
        ["-isystem " + d for d in nccl_cc.system_includes.to_list()],
    )

    out = ctx.actions.declare_file("libnccl_device.bc")
    ctx.actions.run_shell(
        inputs = (
            ctx.files.srcs + nccl_headers +
            ctx.files._cuda_nvcc + cuda_header_files +
            ctx.files._clang_headers +
            [ctx.file._compile_script]
        ),
        outputs = [out],
        tools = [
            ctx.executable._clang,
            ctx.executable._opt,
            ctx.executable._llvm_dis,
            ctx.executable._llvm_as,
        ],
        command = """
            bash {script} \
                --clang {clang} \
                --opt {opt} \
                --llvm-dis {llvm_dis} \
                --llvm-as {llvm_as} \
                --cuda-nvcc-path {cuda_nvcc_path} \
                --cuda-inc-dirs '{cuda_inc_dirs}' \
                --resource-dir {resource_dir} \
                --gpu-arch {gpu_arch} \
                --nccl-device-header {nccl_device_h} \
                --nccl-inc-flags '{nccl_inc_flags}' \
                --src {src} \
                --out {out}
        """.format(
            script = ctx.file._compile_script.path,
            clang = ctx.executable._clang.path,
            opt = ctx.executable._opt.path,
            llvm_dis = ctx.executable._llvm_dis.path,
            llvm_as = ctx.executable._llvm_as.path,
            cuda_nvcc_path = cuda_nvcc_path,
            cuda_inc_dirs = " ".join([d for d in cuda_include_dirs]),
            resource_dir = resource_dir,
            gpu_arch = ctx.attr.gpu_arch,
            nccl_device_h = nccl_device_h.path,
            nccl_inc_flags = nccl_inc_flags,
            src = wrapper_impl.path,
            out = out.path,
        ),
        progress_message = "Building NCCL device bitcode",
        mnemonic = "NcclBitcode",
    )

    return [DefaultInfo(files = depset([out]))]

nccl_device_bitcode = rule(
    implementation = _nccl_device_bitcode_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "nccl_hdrs": attr.label_list(),
        "gpu_arch": attr.string(default = "sm_90"),
        "_compile_script": attr.label(
            default = Label("//third_party/nccl_device_bitcode:compile_bitcode.sh"),
            allow_single_file = True,
        ),
        "_cuda_hdrs": attr.label(
            default = Label("@local_config_cuda//cuda:cuda_headers"),
        ),
        "_cuda_nvcc": attr.label(
            default = Label("@cuda_nvcc//:nvvm"),
        ),
        "_clang": attr.label(
            default = Label("@llvm-project//clang:clang"),
            executable = True, cfg = "exec",
        ),
        "_clang_headers": attr.label(
            default = Label("@llvm-project//clang:builtin_headers_gen"),
            cfg = "exec",
        ),
        "_opt": attr.label(
            default = Label("@llvm-project//llvm:opt"),
            executable = True, cfg = "exec",
        ),
        "_llvm_dis": attr.label(
            default = Label("@llvm-project//llvm:llvm-dis"),
            executable = True, cfg = "exec",
        ),
        "_llvm_as": attr.label(
            default = Label("@llvm-project//llvm:llvm-as"),
            executable = True, cfg = "exec",
        ),
    },
)
