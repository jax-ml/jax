/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/gpu/hybrid_kernels.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

// This helper class is used to define a host buffer that can be copied to and
// from a device buffer.
template <typename T>
class HostBuffer {
 public:
  HostBuffer(std::size_t size) : size_(size) {
    data_ = std::unique_ptr<T[]>(new T[size]);
  }

  absl::Status CopyFromDevice(gpuStream_t stream, const T* buffer) {
    return JAX_AS_STATUS(gpuMemcpyAsync(data_.get(), buffer, size_ * sizeof(T),
                                        gpuMemcpyDeviceToHost, stream));
  }

  absl::Status CopyToDevice(gpuStream_t stream, T* buffer) {
    return JAX_AS_STATUS(gpuMemcpyAsync(buffer, data_.get(), size_ * sizeof(T),
                                        gpuMemcpyHostToDevice, stream));
  }

  T* get() const { return data_.get(); }

 private:
  std::unique_ptr<T[]> data_;
  size_t size_;
};

// Forwarded from MAGMA for use as an input parameter.
typedef enum {
  MagmaNoVec = 301,
  MagmaVec = 302,
} magma_vec_t;

// Compile time lookup of MAGMA function names depending on the data type.
template <ffi::DataType DataType>
struct always_false : std::false_type {};

template <ffi::DataType DataType>
struct MagmaGeev {
  static_assert(always_false<DataType>::value, "unsupported data type");
};
template <>
struct MagmaGeev<ffi::F32> {
  static constexpr char name[] = "magma_sgeev";
};
template <>
struct MagmaGeev<ffi::F64> {
  static constexpr char name[] = "magma_dgeev";
};
template <>
struct MagmaGeev<ffi::C64> {
  static constexpr char name[] = "magma_cgeev";
};
template <>
struct MagmaGeev<ffi::C128> {
  static constexpr char name[] = "magma_zgeev";
};
template <ffi::DataType DataType>
struct MagmaGeqp3 {
  static_assert(always_false<DataType>::value, "unsupported data type");
};
template <>
struct MagmaGeqp3<ffi::F32> {
  static constexpr char name[] = "magma_sgeqp3_gpu";
  static constexpr char expert_name[] = "magma_sgeqp3_expert_gpu_work";
  static constexpr char block_size_name[] = "magma_get_sgeqp3_nb";
};
template <>
struct MagmaGeqp3<ffi::F64> {
  static constexpr char name[] = "magma_dgeqp3_gpu";
  static constexpr char expert_name[] = "magma_dgeqp3_expert_gpu_work";
  static constexpr char block_size_name[] = "magma_get_dgeqp3_nb";
};
template <>
struct MagmaGeqp3<ffi::C64> {
  static constexpr char name[] = "magma_cgeqp3_gpu";
  static constexpr char expert_name[] = "magma_cgeqp3_expert_gpu_work";
  static constexpr char block_size_name[] = "magma_get_cgeqp3_nb";
};
template <>
struct MagmaGeqp3<ffi::C128> {
  static constexpr char name[] = "magma_zgeqp3_gpu";
  static constexpr char expert_name[] = "magma_zgeqp3_expert_gpu_work";
  static constexpr char block_size_name[] = "magma_get_zgeqp3_nb";
};

MagmaLookup::~MagmaLookup() {
  if (initialized_) {
    void* magma_finalize = dlsym(handle_, "magma_finalize");
    if (magma_finalize != nullptr) {
      reinterpret_cast<void (*)()>(magma_finalize)();
    }
  }
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
}

absl::StatusOr<void*> MagmaLookup::FindMagmaInit() {
  void* magma_init = nullptr;
  std::vector<const char*> paths;
  const char* magma_lib_path = std::getenv("JAX_GPU_MAGMA_PATH");
  if (magma_lib_path != nullptr) {
    paths.push_back(magma_lib_path);
  } else {
    paths.push_back("libmagma.so.2");
    paths.push_back("libmagma.so");
    paths.push_back(nullptr);
  }
  for (const auto& path : paths) {
    handle_ = dlopen(path, RTLD_LAZY);
    if (handle_ != nullptr) {
      magma_init = dlsym(handle_, "magma_init");
      if (magma_init != nullptr) {
        if (path != nullptr) {
          lib_path_ = std::string(path);
        }
        break;
      }
    }
  }
  if (handle_ == nullptr || magma_init == nullptr) {
    return absl::InternalError(
        "Unable to dlopen a MAGMA shared library that defines a magma_init "
        "symbol. Use the JAX_GPU_MAGMA_PATH environment variable to "
        "specify an explicit path to the library.");
  }
  return magma_init;
}

absl::Status MagmaLookup::Initialize() {
  if (failed_) {
    return absl::InternalError("MAGMA initialization was unsuccessful.");
  }
  if (!initialized_) {
    auto maybe_magma_init = FindMagmaInit();
    if (!maybe_magma_init.ok()) {
      failed_ = true;
      return maybe_magma_init.status();
    }
    reinterpret_cast<void (*)()>(maybe_magma_init.value())();
    initialized_ = true;
  }
  return absl::OkStatus();
}

absl::StatusOr<void*> MagmaLookup::Find(const char name[]) {
  if (!initialized_) {
    return absl::InternalError("MAGMA support has not been initialized.");
  }

  auto it = symbols_.find(name);
  if (it != symbols_.end()) return it->second;

  void* symbol = dlsym(handle_, name);
  if (symbol == nullptr) {
    if (lib_path_.has_value()) {
      return absl::InternalError(absl::StrFormat(
          "Unable to load the symbol '%s' from the MAGMA library at '%s'.",
          name, lib_path_.value()));

    } else {
      return absl::InternalError(absl::StrFormat(
          "Unable to load a globally defined symbol called '%s'. Use the "
          "JAX_GPU_MAGMA_PATH environment variable to specify an explicit "
          "path to the library.",
          name));
    }
  }

  symbols_.insert({name, symbol});
  return symbol;
}

// Lookup the MAGMA symbol for the given function name. This function only
// dlopen the MAGMA library once per process.
absl::StatusOr<void*> FindMagmaSymbol(const char name[]) {
  static absl::Mutex mu;
  static MagmaLookup& lookup = *new MagmaLookup ABSL_GUARDED_BY(mu);
  absl::MutexLock lock(mu);
  auto status = lookup.Initialize();
  if (!status.ok()) {
    return status;
  }
  return lookup.Find(name);
}

// Column Pivoting QR Factorization

// magma geqp3_gpu

template <ffi::DataType DataType, typename IntType>
class PivotingQrFactorizationHost {
  using RealType = ffi::NativeType<ffi::ToReal(DataType)>;
  using ValueType = ffi::NativeType<DataType>;

 public:
  explicit PivotingQrFactorizationHost() = default;
  PivotingQrFactorizationHost(PivotingQrFactorizationHost&&) = default;

  ffi::Error compute(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer x, ffi::AnyBuffer jpvt,
                     ffi::Result<ffi::AnyBuffer> x_out,
                     ffi::Result<ffi::AnyBuffer> jpvt_out,
                     ffi::Result<ffi::AnyBuffer> tau) {
    FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<IntType>(rows));
    FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<IntType>(cols));
    auto min_dim = std::min(m, n);

    FFI_ASSIGN_OR_RETURN(IntType lwork, lwork(m, n));
    auto work = AllocateScratchMemory<DataType>(lwork);

    constexpr bool is_complex_dtype = ffi::IsComplexType<DataType>();
    std::unique_ptr<RealType[]> rwork;
    if constexpr (is_complex_dtype) {
      rwork = AllocateScratchMemory<ffi::ToReal(DataType)>(2 * n);
    }

    auto x_host = HostBuffer<ValueType>(x.element_count());
    FFI_RETURN_IF_ERROR_STATUS(
        x_host.CopyFromDevice(stream, x.typed_data<ValueType>()));
    auto jpvt_host = HostBuffer<int32_t>(jpvt.element_count());
    FFI_RETURN_IF_ERROR_STATUS(
        jpvt_host.CopyFromDevice(stream, jpvt.typed_data<int32_t>()));
    auto tau_host = HostBuffer<ValueType>(batch * min_dim);
    auto info_host = HostBuffer<int32_t>(batch);

    std::unique_ptr<IntType[]> jpvt_tmp;
    if constexpr (!std::is_same_v<IntType, int32_t>) {
      jpvt_tmp = std::make_unique<IntType[]>(n);
    }

    for (int64_t i = 0; i < batch; ++i) {
      IntType info_v;
      if constexpr (std::is_same_v<IntType, int32_t>) {
        if constexpr (is_complex_dtype) {
          PivotingQrFactorization<DataType, IntType>::fn(
              &m, &n, x_host.get() + i * m * n, &m, jpvt_host.get() + i * n,
              tau_host.get() + i * min_dim, work.get(), &lwork, rwork.get(),
              &info_v);
        } else {
          PivotingQrFactorization<DataType, IntType>::fn(
              &m, &n, x_host.get() + i * m * n, &m, jpvt_host.get() + i * n,
              tau_host.get() + i * min_dim, work.get(), &lwork, &info_v);
        }
      } else {
        for (int64_t j = 0; j < n; ++j) {
          jpvt_tmp[j] = static_cast<IntType>(jpvt_host.get()[i * n + j]);
        }
        if constexpr (is_complex_dtype) {
          PivotingQrFactorization<DataType, IntType>::fn(
              &m, &n, x_host.get() + i * m * n, &m, jpvt_tmp.get(),
              tau_host.get() + i * min_dim, work.get(), &lwork, rwork.get(),
              &info_v);
        } else {
          PivotingQrFactorization<DataType, IntType>::fn(
              &m, &n, x_host.get() + i * m * n, &m, jpvt_tmp.get(),
              tau_host.get() + i * min_dim, work.get(), &lwork, &info_v);
        }
        for (int64_t j = 0; j < n; ++j) {
          jpvt_host.get()[i * n + j] = static_cast<int32_t>(jpvt_tmp[j]);
        }
      }
      info_host.get()[i] = static_cast<int32_t>(info_v);
    }

    FFI_RETURN_IF_ERROR_STATUS(
        x_host.CopyToDevice(stream, x_out->typed_data<ValueType>()));
    FFI_RETURN_IF_ERROR_STATUS(
        jpvt_host.CopyToDevice(stream, jpvt_out->typed_data<int>()));
    FFI_RETURN_IF_ERROR_STATUS(
        tau_host.CopyToDevice(stream, tau->typed_data<ValueType>()));
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));
    return ffi::Error::Success();
  }

 private:
  absl::StatusOr<IntType> lwork(IntType m, IntType n) {
    int64_t lwork =
        PivotingQrFactorization<DataType, IntType>::GetWorkspaceSize(m, n);
    return MaybeCastNoOverflow<IntType>(lwork);
  }
};

template <ffi::DataType DataType>
class PivotingQrFactorizationMagma {
  using RealType = ffi::NativeType<ffi::ToReal(DataType)>;
  using ValueType = ffi::NativeType<DataType>;
  using Fn = std::conditional_t<
      ffi::IsComplexType<DataType>(),
      int32_t(int32_t m, int32_t n, ValueType* dA, int32_t ldda, int32_t* jpvt,
              ValueType* tau, ValueType* dwork, int32_t lwork, RealType* rwork,
              int32_t* info),
      int32_t(int32_t m, int32_t n, RealType* dA, int32_t ldda, int32_t* jpvt,
              RealType* tau, RealType* dwork, int32_t lwork, int32_t* info)>;
  using BlockSizeFn = int32_t(int32_t m, int32_t n);

 public:
  explicit PivotingQrFactorizationMagma() = default;
  PivotingQrFactorizationMagma(PivotingQrFactorizationMagma&&) = default;

  ffi::Error compute(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer x, ffi::AnyBuffer jpvt,
                     ffi::Result<ffi::AnyBuffer> x_out,
                     ffi::Result<ffi::AnyBuffer> jpvt_out,
                     ffi::Result<ffi::AnyBuffer> tau) {
    FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int32_t>(rows));
    FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int32_t>(cols));
    auto min_dim = std::min(m, n);

    FFI_ASSIGN_OR_RETURN(int32_t lwork, lwork(m, n));
    FFI_ASSIGN_OR_RETURN(auto work,
                         AllocateWorkspace<ValueType>(scratch, lwork, "geqp3"));

    constexpr bool is_complex_dtype = ffi::IsComplexType<DataType>();
    RealType* rwork;
    if constexpr (is_complex_dtype) {
      FFI_ASSIGN_OR_RETURN(
          rwork, AllocateWorkspace<RealType>(scratch, 2 * n, "geqp3"));
    }

    auto x_data = x.typed_data<ValueType>();
    auto x_out_data = x_out->typed_data<ValueType>();
    auto tau_data = tau->typed_data<ValueType>();
    if (x_data != x_out_data) {
      FFI_RETURN_IF_ERROR_STATUS(
          JAX_AS_STATUS(gpuMemcpyAsync(x_out_data, x_data, x.size_bytes(),
                                       gpuMemcpyDeviceToDevice, stream)));
    }
    auto jpvt_host = HostBuffer<int32_t>(jpvt.element_count());
    FFI_RETURN_IF_ERROR_STATUS(
        jpvt_host.CopyFromDevice(stream, jpvt.typed_data<int32_t>()));
    auto info_host = HostBuffer<int32_t>(batch);

    // TODO: do we need to wrap with synchronise due to non-stream safety.
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));
    for (int64_t i = 0; i < batch; ++i) {
      if constexpr (is_complex_dtype) {
        fn_(m, n, x_out_data + i * m * n, m, jpvt_host.get() + i * n,
            tau_data + i * min_dim, work, lwork, rwork, info_host.get() + i);
      } else {
        fn_(m, n, x_out_data + i * m * n, m, jpvt_host.get() + i * n,
            tau_data + i * min_dim, work, lwork, info_host.get() + i);
      }
    }
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));
    FFI_RETURN_IF_ERROR_STATUS(
        jpvt_host.CopyToDevice(stream, jpvt_out->typed_data<int32_t>()));
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));
    return ffi::Error::Success();
  }

 private:
  Fn* fn_ = nullptr;

  absl::StatusOr<int32_t> lwork(int32_t m, int32_t n) {
    // `{c,d,s,z}_geqp3_gpu` do not support a workspace query, but we can call
    // the expert API instead.
    auto maybe_ptr = FindMagmaSymbol(MagmaGeqp3<DataType>::name);
    if (!maybe_ptr.ok()) return maybe_ptr.status();
    fn_ = reinterpret_cast<Fn*>(*maybe_ptr);

    auto maybe_expert_ptr = FindMagmaSymbol(MagmaGeqp3<DataType>::expert_name);
    if (!maybe_expert_ptr.ok()) return maybe_expert_ptr.status();
    using ExpertFn = int32_t(int32_t m, int32_t n, ValueType* dA, int32_t ldda,
                             int32_t* jpvt, ValueType* tau, void* host_work,
                             int32_t* lwork_host, void* device_work,
                             int32_t* lwork_device, int32_t* info, void* queue);
    auto* expert_fn = reinterpret_cast<ExpertFn*>(*maybe_expert_ptr);

    int32_t lwork_host = -1;
    int32_t lwork_device = -1;
    int32_t info = 0;
    expert_fn(m, n, nullptr, std::max(1, static_cast<int>(m)), nullptr, nullptr,
              nullptr, &lwork_host, nullptr, &lwork_device, &info, nullptr);
    if (info != 0) {
      return absl::InternalError(absl::StrFormat(
          "MAGMA expert geqp3 workspace query failed with info=%d", info));
    }
    return lwork_device / sizeof(ValueType);
  }
};

ffi::Error PivotingQrFactorizationDispatch(
    gpuStream_t stream, ffi::ScratchAllocator scratch, std::string_view magma,
    ffi::AnyBuffer x, ffi::AnyBuffer jpvt, ffi::Result<ffi::AnyBuffer> x_out,
    ffi::Result<ffi::AnyBuffer> jpvt_out, ffi::Result<ffi::AnyBuffer> tau) {
  auto dataType = x.element_type();
  if (dataType != x_out->element_type() || dataType != tau->element_type()) {
    return ffi::Error::InvalidArgument(
        "The buffers 'x', 'x_out' and 'tau' must have the same element type.");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(x.dimensions()));
  FFI_RETURN_IF_ERROR(
      CheckShape(jpvt.dimensions(), {batch, cols}, "jpvt", "geqp3"));
  FFI_RETURN_IF_ERROR(
      CheckShape(x_out->dimensions(), {batch, rows, cols}, "x_out", "geqp3"));
  FFI_RETURN_IF_ERROR(
      CheckShape(jpvt_out->dimensions(), {batch, cols}, "jpvt_out", "geqp3"));
  FFI_RETURN_IF_ERROR(CheckShape(
      tau->dimensions(), {batch, std::min(rows, cols)}, "tau", "geqp3"));

  bool use_magma = magma == "on";
  if (magma == "auto" && cols >= 2048) {
    use_magma = FindMagmaSymbol("magma_init").ok();
  }

  switch (dataType) {
    case ffi::F32:
      if (use_magma) {
        return PivotingQrFactorizationMagma<ffi::F32>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      } else {
        if (PivotingQrFactorization<ffi::F32, int64_t>::fn != nullptr) {
          return PivotingQrFactorizationHost<ffi::F32, int64_t>().compute(
              batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out,
              tau);
        }
        return PivotingQrFactorizationHost<ffi::F32, int32_t>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      }
    case ffi::F64:
      if (use_magma) {
        return PivotingQrFactorizationMagma<ffi::F64>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      } else {
        if (PivotingQrFactorization<ffi::F64, int64_t>::fn != nullptr) {
          return PivotingQrFactorizationHost<ffi::F64, int64_t>().compute(
              batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out,
              tau);
        }
        return PivotingQrFactorizationHost<ffi::F64, int32_t>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      }
    case ffi::C64:
      if (use_magma) {
        return PivotingQrFactorizationMagma<ffi::C64>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      } else {
        if (PivotingQrFactorization<ffi::C64, int64_t>::fn != nullptr) {
          return PivotingQrFactorizationHost<ffi::C64, int64_t>().compute(
              batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out,
              tau);
        }
        return PivotingQrFactorizationHost<ffi::C64, int32_t>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      }
    case ffi::C128:
      if (use_magma) {
        return PivotingQrFactorizationMagma<ffi::C128>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      } else {
        if (PivotingQrFactorization<ffi::C128, int64_t>::fn != nullptr) {
          return PivotingQrFactorizationHost<ffi::C128, int64_t>().compute(
              batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out,
              tau);
        }
        return PivotingQrFactorizationHost<ffi::C128, int32_t>().compute(
            batch, rows, cols, stream, scratch, x, jpvt, x_out, jpvt_out, tau);
      }
    default:
      return ffi::Error::InvalidArgument(absl::StrFormat(
          "Unsupported dtype %s in geqp3", absl::FormatStreamed(dataType)));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kGeqp3, PivotingQrFactorizationDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<std::string_view>("magma")
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Arg<ffi::AnyBuffer>()  // jpvt
                                  .Ret<ffi::AnyBuffer>()  // x_out
                                  .Ret<ffi::AnyBuffer>()  // jpvt_out
                                  .Ret<ffi::AnyBuffer>()  // tau
);

// Real-valued eigendecomposition

template <ffi::DataType DataType, typename IntType>
class EigRealHost {
  using Real = ffi::NativeType<DataType>;

 public:
  explicit EigRealHost() = default;
  EigRealHost(EigRealHost&&) = default;

  absl::StatusOr<IntType> lwork(IntType n, bool left, bool right) {
    n_ = n;
    jobvl_ = left ? 'V' : 'N';
    jobvr_ = right ? 'V' : 'N';
    int64_t lwork =
        EigenvalueDecomposition<DataType, IntType>::GetWorkspaceSize(
            n, static_cast<eig::ComputationMode>(jobvl_),
            static_cast<eig::ComputationMode>(jobvr_));
    return MaybeCastNoOverflow<IntType>(lwork);
  }

  void compute(Real* x, Real* wr, Real* wi, Real* vl, Real* vr, Real* work,
               IntType lwork, IntType* info) {
    EigenvalueDecomposition<DataType, IntType>::fn(&jobvl_, &jobvr_, &n_, x,
                                                   &n_, wr, wi, vl, &n_, vr,
                                                   &n_, work, &lwork, info);
  }

 private:
  IntType n_;
  char jobvl_, jobvr_;
};

template <ffi::DataType DataType>
class EigRealMagma {
  using Real = ffi::NativeType<DataType>;
  using Fn = int32_t(magma_vec_t, magma_vec_t, int32_t, Real*, int32_t, Real*,
                     Real*, Real*, int32_t, Real*, int32_t, Real*, int32_t,
                     int32_t*);

 public:
  explicit EigRealMagma() = default;
  EigRealMagma(EigRealMagma&&) = default;

  absl::StatusOr<int32_t> lwork(int32_t n, bool left, bool right) {
    n_ = n;
    jobvl_ = left ? MagmaVec : MagmaNoVec;
    jobvr_ = right ? MagmaVec : MagmaNoVec;

    auto maybe_ptr = FindMagmaSymbol(MagmaGeev<DataType>::name);
    if (!maybe_ptr.ok()) return maybe_ptr.status();
    fn_ = reinterpret_cast<Fn*>(*maybe_ptr);

    int32_t query_info;
    Real query_host;
    fn_(jobvl_, jobvr_, n, nullptr, n, nullptr, nullptr, nullptr, n, nullptr, n,
        &query_host, -1, &query_info);
    return static_cast<int32_t>(query_host);
  }

  void compute(Real* x, Real* wr, Real* wi, Real* vl, Real* vr, Real* work,
               int32_t lwork, int32_t* info) {
    fn_(jobvl_, jobvr_, n_, x, n_, wr, wi, vl, n_, vr, n_, work, lwork, info);
  }

 private:
  int32_t n_;
  magma_vec_t jobvl_, jobvr_;
  Fn* fn_ = nullptr;
};

template <ffi::DataType DataType, typename IntType, typename Impl>
ffi::Error EigReal(Impl impl, int64_t batch, int64_t cols, gpuStream_t stream,
                   bool left, bool right, ffi::AnyBuffer x,
                   ffi::Result<ffi::AnyBuffer> wr,
                   ffi::Result<ffi::AnyBuffer> wi,
                   ffi::Result<ffi::AnyBuffer> vl,
                   ffi::Result<ffi::AnyBuffer> vr,
                   ffi::Result<ffi::Buffer<ffi::S32>> info) {
  using Real = ffi::NativeType<DataType>;
  using Complex = ffi::NativeType<ffi::ToComplex(DataType)>;

  auto x_host = HostBuffer<Real>(x.element_count());
  FFI_RETURN_IF_ERROR_STATUS(
      x_host.CopyFromDevice(stream, x.typed_data<Real>()));

  auto wr_host = HostBuffer<Real>(batch * cols);
  auto wi_host = HostBuffer<Real>(batch * cols);
  auto vl_host = HostBuffer<Complex>(batch * cols * cols);
  auto vr_host = HostBuffer<Complex>(batch * cols * cols);
  auto info_host = HostBuffer<IntType>(batch);

  FFI_ASSIGN_OR_RETURN(IntType n, MaybeCastNoOverflow<IntType>(cols));
  FFI_ASSIGN_OR_RETURN(IntType lwork, impl.lwork(n, left, right));
  auto work_host = AllocateScratchMemory<DataType>(lwork);
  auto work_left = AllocateScratchMemory<DataType>(cols * cols);
  auto work_right = AllocateScratchMemory<DataType>(cols * cols);

  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  const auto is_finite = [](auto* data, int64_t size) {
    return absl::c_all_of(absl::MakeSpan(data, size),
                          [](auto value) { return std::isfinite(value); });
  };

  for (int64_t i = 0; i < batch; ++i) {
    if (is_finite(x_host.get() + i * cols * cols, cols * cols)) {
      impl.compute(x_host.get() + i * cols * cols, wr_host.get() + i * cols,
                   wi_host.get() + i * cols, work_left.get(), work_right.get(),
                   work_host.get(), lwork, info_host.get() + i);
      if (info_host.get()[i] == 0) {
        if (left) {
          UnpackEigenvectors(n, wi_host.get() + i * cols, work_left.get(),
                             vl_host.get() + i * cols * cols);
        }
        if (right) {
          UnpackEigenvectors(n, wi_host.get() + i * cols, work_right.get(),
                             vr_host.get() + i * cols * cols);
        }
      }
    } else {
      info_host.get()[i] = -4;
    }
  }

  auto info_host_out = HostBuffer<int32_t>(batch);
  for (int64_t i = 0; i < batch; ++i) {
    info_host_out.get()[i] = static_cast<int32_t>(info_host.get()[i]);
  }
  FFI_RETURN_IF_ERROR_STATUS(
      wr_host.CopyToDevice(stream, wr->typed_data<Real>()));
  FFI_RETURN_IF_ERROR_STATUS(
      wi_host.CopyToDevice(stream, wi->typed_data<Real>()));
  FFI_RETURN_IF_ERROR_STATUS(
      info_host_out.CopyToDevice(stream, info->typed_data()));
  if (left) {
    FFI_RETURN_IF_ERROR_STATUS(
        vl_host.CopyToDevice(stream, vl->typed_data<Complex>()));
  }
  if (right) {
    FFI_RETURN_IF_ERROR_STATUS(
        vr_host.CopyToDevice(stream, vr->typed_data<Complex>()));
  }
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  return ffi::Error::Success();
}

ffi::Error EigRealDispatch(gpuStream_t stream, std::string_view magma,
                           bool left, bool right, ffi::AnyBuffer x,
                           ffi::Result<ffi::AnyBuffer> wr,
                           ffi::Result<ffi::AnyBuffer> wi,
                           ffi::Result<ffi::AnyBuffer> vl,
                           ffi::Result<ffi::AnyBuffer> vr,
                           ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = x.element_type();
  if (dataType != wr->element_type() || dataType != wi->element_type() ||
      ffi::ToComplex(dataType) != vl->element_type() ||
      ffi::ToComplex(dataType) != vr->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to eig must have the same element type");
  }

  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(x.dimensions()));
  if (rows != cols) {
    return ffi::Error::InvalidArgument(
        "The input matrix to eig must be square");
  }
  FFI_RETURN_IF_ERROR(CheckShape(wr->dimensions(), {batch, cols}, "wr", "eig"));
  FFI_RETURN_IF_ERROR(CheckShape(wi->dimensions(), {batch, cols}, "wi", "eig"));
  if (left) {
    FFI_RETURN_IF_ERROR(
        CheckShape(vl->dimensions(), {batch, rows, cols}, "vl", "eig"));
  }
  if (right) {
    FFI_RETURN_IF_ERROR(
        CheckShape(vr->dimensions(), {batch, rows, cols}, "vr", "eig"));
  }
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "eig"));

  bool use_magma = magma == "on";
  if (magma == "auto" && cols >= 2048) {
    use_magma = FindMagmaSymbol("magma_init").ok();
  }

  switch (dataType) {
    case ffi::F32:
      if (use_magma) {
        return EigReal<ffi::F32, int32_t>(EigRealMagma<ffi::F32>(), batch, cols,
                                          stream, left, right, x, wr, wi, vl,
                                          vr, info);
      } else {
        if (EigenvalueDecomposition<ffi::F32, int64_t>::fn != nullptr) {
          return EigReal<ffi::F32, int64_t>(EigRealHost<ffi::F32, int64_t>(),
                                            batch, cols, stream, left, right, x,
                                            wr, wi, vl, vr, info);
        }
        return EigReal<ffi::F32, int32_t>(EigRealHost<ffi::F32, int32_t>(),
                                          batch, cols, stream, left, right, x,
                                          wr, wi, vl, vr, info);
      }
    case ffi::F64:
      if (use_magma) {
        return EigReal<ffi::F64, int32_t>(EigRealMagma<ffi::F64>(), batch, cols,
                                          stream, left, right, x, wr, wi, vl,
                                          vr, info);
      } else {
        if (EigenvalueDecomposition<ffi::F64, int64_t>::fn != nullptr) {
          return EigReal<ffi::F64, int64_t>(EigRealHost<ffi::F64, int64_t>(),
                                            batch, cols, stream, left, right, x,
                                            wr, wi, vl, vr, info);
        }
        return EigReal<ffi::F64, int32_t>(EigRealHost<ffi::F64, int32_t>(),
                                          batch, cols, stream, left, right, x,
                                          wr, wi, vl, vr, info);
      }
    default:
      return ffi::Error::InvalidArgument(absl::StrFormat(
          "Unsupported dtype %s in eig_real", absl::FormatStreamed(dataType)));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kEigReal, EigRealDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Attr<std::string_view>("magma")
                                  .Attr<bool>("left")
                                  .Attr<bool>("right")
                                  .Arg<ffi::AnyBuffer>()         // x
                                  .Ret<ffi::AnyBuffer>()         // wr
                                  .Ret<ffi::AnyBuffer>()         // wi
                                  .Ret<ffi::AnyBuffer>()         // vl
                                  .Ret<ffi::AnyBuffer>()         // vr
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

// Complex-valued eigendecomposition

template <ffi::DataType DataType, typename IntType>
class EigCompHost {
  using Real = ffi::NativeType<ffi::ToReal(DataType)>;
  using Complex = ffi::NativeType<DataType>;

 public:
  explicit EigCompHost() = default;
  EigCompHost(EigCompHost&&) = default;

  absl::StatusOr<IntType> lwork(IntType n, bool left, bool right) {
    n_ = n;
    jobvl_ = left ? 'V' : 'N';
    jobvr_ = right ? 'V' : 'N';
    int64_t lwork =
        EigenvalueDecompositionComplex<DataType, IntType>::GetWorkspaceSize(
            n, static_cast<eig::ComputationMode>(jobvl_),
            static_cast<eig::ComputationMode>(jobvr_));
    return MaybeCastNoOverflow<IntType>(lwork);
  }

  void compute(Complex* x, Complex* w, Complex* vl, Complex* vr, Complex* work,
               IntType lwork, Real* rwork, IntType* info) {
    EigenvalueDecompositionComplex<DataType, IntType>::fn(
        &jobvl_, &jobvr_, &n_, x, &n_, w, vl, &n_, vr, &n_, work, &lwork, rwork,
        info);
  }

 private:
  IntType n_;
  char jobvl_, jobvr_;
};

template <ffi::DataType DataType>
class EigCompMagma {
  using Real = ffi::NativeType<ffi::ToReal(DataType)>;
  using Complex = ffi::NativeType<DataType>;
  using Fn = int32_t(magma_vec_t, magma_vec_t, int32_t, Complex*, int32_t,
                     Complex*, Complex*, int32_t, Complex*, int32_t, Complex*,
                     int32_t, Real*, int32_t*);

 public:
  explicit EigCompMagma() = default;
  EigCompMagma(EigCompMagma&&) = default;

  absl::StatusOr<int32_t> lwork(int32_t n, bool left, bool right) {
    n_ = n;
    jobvl_ = left ? MagmaVec : MagmaNoVec;
    jobvr_ = right ? MagmaVec : MagmaNoVec;
    lda_ = std::max(n_, 1);
    ldvl_ = left ? n_ : 1;
    ldvr_ = right ? n_ : 1;

    auto maybe_ptr = FindMagmaSymbol(MagmaGeev<DataType>::name);
    if (!maybe_ptr.ok()) return maybe_ptr.status();
    fn_ = reinterpret_cast<Fn*>(*maybe_ptr);

    int32_t query_info;
    Complex query_host;
    fn_(jobvl_, jobvr_, n_, nullptr, lda_, nullptr, nullptr, ldvl_, nullptr,
        ldvr_, &query_host, -1, nullptr, &query_info);
    return static_cast<int32_t>(query_host.real());
  }

  void compute(Complex* x, Complex* w, Complex* vl, Complex* vr, Complex* work,
               int32_t lwork, Real* rwork, int32_t* info) {
    fn_(jobvl_, jobvr_, n_, x, lda_, w, vl, ldvl_, vr, ldvr_, work, lwork,
        rwork, info);
  }

 private:
  int32_t n_, lda_, ldvl_, ldvr_;
  magma_vec_t jobvl_, jobvr_;
  Fn* fn_ = nullptr;
};

template <ffi::DataType DataType, typename IntType, typename Impl>
ffi::Error EigComp(Impl impl, int64_t batch, int64_t cols, gpuStream_t stream,
                   bool left, bool right, ffi::AnyBuffer x,
                   ffi::Result<ffi::AnyBuffer> w,
                   ffi::Result<ffi::AnyBuffer> vl,
                   ffi::Result<ffi::AnyBuffer> vr,
                   ffi::Result<ffi::Buffer<ffi::S32>> info) {
  using Complex = ffi::NativeType<DataType>;

  auto x_host = HostBuffer<Complex>(x.element_count());
  FFI_RETURN_IF_ERROR_STATUS(
      x_host.CopyFromDevice(stream, x.typed_data<Complex>()));

  auto w_host = HostBuffer<Complex>(batch * cols);
  auto vl_host = HostBuffer<Complex>(batch * cols * cols);
  auto vr_host = HostBuffer<Complex>(batch * cols * cols);
  auto info_host = HostBuffer<IntType>(batch);

  FFI_ASSIGN_OR_RETURN(IntType n, MaybeCastNoOverflow<IntType>(cols));
  FFI_ASSIGN_OR_RETURN(IntType lwork, impl.lwork(n, left, right));
  auto work_host = AllocateScratchMemory<DataType>(lwork);
  auto rwork_host =
      AllocateScratchMemory<ffi::ToReal(DataType)>(2 * cols * cols);

  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  const auto is_finite = [](auto* data, int64_t size) {
    return absl::c_all_of(absl::MakeSpan(data, size), [](const auto& z) {
      return std::isfinite(z.real()) && std::isfinite(z.imag());
    });
  };

  for (int64_t i = 0; i < batch; ++i) {
    if (is_finite(x_host.get() + i * cols * cols, cols * cols)) {
      impl.compute(x_host.get() + i * cols * cols, w_host.get() + i * cols,
                   vl_host.get() + i * cols * cols,
                   vr_host.get() + i * cols * cols, work_host.get(), lwork,
                   rwork_host.get(), info_host.get() + i);
    } else {
      info_host.get()[i] = -4;
    }
  }

  FFI_RETURN_IF_ERROR_STATUS(
      w_host.CopyToDevice(stream, w->typed_data<Complex>()));
  if (left) {
    FFI_RETURN_IF_ERROR_STATUS(
        vl_host.CopyToDevice(stream, vl->typed_data<Complex>()));
  }
  if (right) {
    FFI_RETURN_IF_ERROR_STATUS(
        vr_host.CopyToDevice(stream, vr->typed_data<Complex>()));
  }
  auto info_host_out = HostBuffer<int32_t>(batch);
  for (int64_t i = 0; i < batch; ++i) {
    info_host_out.get()[i] = static_cast<int32_t>(info_host.get()[i]);
  }
  FFI_RETURN_IF_ERROR_STATUS(
      info_host_out.CopyToDevice(stream, info->typed_data()));
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  return ffi::Error::Success();
}

ffi::Error EigCompDispatch(gpuStream_t stream, std::string_view magma,
                           bool left, bool right, ffi::AnyBuffer x,
                           ffi::Result<ffi::AnyBuffer> w,
                           ffi::Result<ffi::AnyBuffer> vl,
                           ffi::Result<ffi::AnyBuffer> vr,
                           ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = x.element_type();
  if (dataType != w->element_type() || dataType != vl->element_type() ||
      dataType != vr->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to eig must have the same element type");
  }

  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(x.dimensions()));
  if (rows != cols) {
    return ffi::Error::InvalidArgument(
        "The input matrix to eig must be square");
  }
  FFI_RETURN_IF_ERROR(CheckShape(w->dimensions(), {batch, cols}, "w", "eig"));
  if (left) {
    FFI_RETURN_IF_ERROR(
        CheckShape(vl->dimensions(), {batch, rows, cols}, "vl", "eig"));
  }
  if (right) {
    FFI_RETURN_IF_ERROR(
        CheckShape(vr->dimensions(), {batch, rows, cols}, "vr", "eig"));
  }
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "eig"));

  bool use_magma = magma == "on";
  if (magma == "auto" && cols >= 2048) {
    use_magma = FindMagmaSymbol("magma_init").ok();
  }

  switch (dataType) {
    case ffi::C64:
      if (use_magma) {
        return EigComp<ffi::C64, int32_t>(EigCompMagma<ffi::C64>(), batch, cols,
                                          stream, left, right, x, w, vl, vr,
                                          info);
      } else {
        if (EigenvalueDecompositionComplex<ffi::C64, int64_t>::fn != nullptr) {
          return EigComp<ffi::C64, int64_t>(EigCompHost<ffi::C64, int64_t>(),
                                            batch, cols, stream, left, right, x,
                                            w, vl, vr, info);
        }
        return EigComp<ffi::C64, int32_t>(EigCompHost<ffi::C64, int32_t>(),
                                          batch, cols, stream, left, right, x,
                                          w, vl, vr, info);
      }
    case ffi::C128:
      if (use_magma) {
        return EigComp<ffi::C128, int32_t>(EigCompMagma<ffi::C128>(), batch,
                                           cols, stream, left, right, x, w, vl,
                                           vr, info);
      } else {
        if (EigenvalueDecompositionComplex<ffi::C128, int64_t>::fn != nullptr) {
          return EigComp<ffi::C128, int64_t>(EigCompHost<ffi::C128, int64_t>(),
                                             batch, cols, stream, left, right,
                                             x, w, vl, vr, info);
        }
        return EigComp<ffi::C128, int32_t>(EigCompHost<ffi::C128, int32_t>(),
                                           batch, cols, stream, left, right, x,
                                           w, vl, vr, info);
      }
    default:
      return ffi::Error::InvalidArgument(absl::StrFormat(
          "Unsupported dtype %s in eig_comp", absl::FormatStreamed(dataType)));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kEigComp, EigCompDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Attr<std::string_view>("magma")
                                  .Attr<bool>("left")
                                  .Attr<bool>("right")
                                  .Arg<ffi::AnyBuffer>()         // x
                                  .Ret<ffi::AnyBuffer>()         // w
                                  .Ret<ffi::AnyBuffer>()         // vl
                                  .Ret<ffi::AnyBuffer>()         // vr
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
