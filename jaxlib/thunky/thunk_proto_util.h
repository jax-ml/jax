/* Copyright 2026 The JAX Authors.

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

#ifndef JAXLIB_THUNKY_THUNK_PROTO_UTIL_H_
#define JAXLIB_THUNKY_THUNK_PROTO_UTIL_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.pb.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/convolution_filter_thunk.pb.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/shaped_slice.pb.h"

namespace xla::gpu {

// Iterates over all BufferAllocationSliceProto instances in a ThunkProto.
template <typename Callback>
absl::Status ForEachBufferSlice(ThunkProto* thunk, Callback&& cb) {
  switch (thunk->impl_case()) {
    case ThunkProto::kKernelThunk: {
      auto* kt = thunk->mutable_kernel_thunk();
      for (int i = 0; i < kt->args_size(); ++i) {
        cb(kt->mutable_args(i));
      }
      return absl::OkStatus();
    }
    case ThunkProto::kCustomKernelThunk: {
      auto* ckt = thunk->mutable_custom_kernel_thunk();
      for (int i = 0; i < ckt->args_size(); ++i) {
        cb(ckt->mutable_args(i)->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kGemmThunk: {
      auto* gt = thunk->mutable_gemm_thunk();
      cb(gt->mutable_lhs_buffer());
      cb(gt->mutable_rhs_buffer());
      cb(gt->mutable_output_buffer());
      if (gt->has_workspace()) cb(gt->mutable_workspace());
      return absl::OkStatus();
    }
    case ThunkProto::kCublasLtMatmulThunk: {
      auto* mt = thunk->mutable_cublas_lt_matmul_thunk();
      cb(mt->mutable_a()->mutable_slice());
      cb(mt->mutable_b()->mutable_slice());
      cb(mt->mutable_c()->mutable_slice());
      cb(mt->mutable_d()->mutable_slice());
      if (mt->has_bias()) cb(mt->mutable_bias()->mutable_slice());
      if (mt->has_aux()) cb(mt->mutable_aux()->mutable_slice());
      if (mt->has_a_scale()) cb(mt->mutable_a_scale()->mutable_slice());
      if (mt->has_b_scale()) cb(mt->mutable_b_scale()->mutable_slice());
      if (mt->has_c_scale()) cb(mt->mutable_c_scale()->mutable_slice());
      if (mt->has_d_scale()) cb(mt->mutable_d_scale()->mutable_slice());
      if (mt->has_d_amax()) cb(mt->mutable_d_amax()->mutable_slice());
      if (mt->has_workspace()) cb(mt->mutable_workspace()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kDynamicSliceFusionThunk: {
      auto* ds = thunk->mutable_dynamic_slice_fusion_thunk();
      for (int i = 0; i < ds->parameter_buffers_size(); ++i) {
        cb(ds->mutable_parameter_buffers(i));
      }
      for (int i = 0; i < ds->result_buffers_size(); ++i) {
        cb(ds->mutable_result_buffers(i));
      }
      return absl::OkStatus();
    }
    case ThunkProto::kMemzeroThunk: {
      cb(thunk->mutable_memzero_thunk()
             ->mutable_dest_buffer()
             ->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kMemset32BitValueThunk: {
      cb(thunk->mutable_memset32bit_value_thunk()->mutable_dest_buffer());
      return absl::OkStatus();
    }
    case ThunkProto::kDeviceToDeviceCopyThunk: {
      auto* ct =
          thunk->mutable_device_to_device_copy_thunk()->mutable_copy_thunk();
      cb(ct->mutable_source_buffer()->mutable_slice());
      cb(ct->mutable_destination_buffer()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kDeviceToHostCopyThunk: {
      auto* ct =
          thunk->mutable_device_to_host_copy_thunk()->mutable_copy_thunk();
      cb(ct->mutable_source_buffer()->mutable_slice());
      cb(ct->mutable_destination_buffer()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kHostToDeviceCopyThunk: {
      auto* ct =
          thunk->mutable_host_to_device_copy_thunk()->mutable_copy_thunk();
      cb(ct->mutable_source_buffer()->mutable_slice());
      cb(ct->mutable_destination_buffer()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kCopyThunk: {
      auto* ct = thunk->mutable_copy_thunk();
      cb(ct->mutable_source_buffer()->mutable_slice());
      cb(ct->mutable_destination_buffer()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kCustomCallThunk: {
      auto* cc = thunk->mutable_custom_call_thunk();
      for (int i = 0; i < cc->operands_size(); ++i) {
        if (cc->mutable_operands(i)->has_shaped_slice()) {
          cb(cc->mutable_operands(i)->mutable_shaped_slice()->mutable_slice());
        }
      }
      for (int i = 0; i < cc->results_size(); ++i) {
        if (cc->mutable_results(i)->has_shaped_slice()) {
          cb(cc->mutable_results(i)->mutable_shaped_slice()->mutable_slice());
        }
      }
      return absl::OkStatus();
    }
    case ThunkProto::kAllReduceThunk: {
      auto* ar = thunk->mutable_all_reduce_thunk();
      for (int i = 0; i < ar->buffers_size(); ++i) {
        cb(ar->mutable_buffers(i)->mutable_source_buffer()->mutable_slice());
        cb(ar->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kAllGatherThunk: {
      auto* ag = thunk->mutable_all_gather_thunk();
      for (int i = 0; i < ag->buffers_size(); ++i) {
        cb(ag->mutable_buffers(i)->mutable_source_buffer()->mutable_slice());
        cb(ag->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kReduceScatterThunk: {
      auto* rs = thunk->mutable_reduce_scatter_thunk();
      for (int i = 0; i < rs->buffers_size(); ++i) {
        cb(rs->mutable_buffers(i)->mutable_source_buffer()->mutable_slice());
        cb(rs->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kAllToAllThunk: {
      auto* a2a = thunk->mutable_all_to_all_thunk();
      for (int i = 0; i < a2a->buffers_size(); ++i) {
        cb(a2a->mutable_buffers(i)->mutable_source_buffer()->mutable_slice());
        cb(a2a->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kRaggedAllToAllThunk: {
      auto* ra2a = thunk->mutable_ragged_all_to_all_thunk();
      for (int i = 0; i < ra2a->buffers_size(); ++i) {
        cb(ra2a->mutable_buffers(i)->mutable_source_buffer()->mutable_slice());
        cb(ra2a->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kCollectivePermuteThunk: {
      auto* cp = thunk->mutable_collective_permute_thunk();
      for (int i = 0; i < cp->buffers_size(); ++i) {
        cb(cp->mutable_buffers(i)->mutable_source_buffer()->mutable_slice());
        cb(cp->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kCollectiveKernelThunk: {
      auto* ck = thunk->mutable_collective_kernel_thunk();
      for (int i = 0; i < ck->buffers_size(); ++i) {
        cb(ck->mutable_buffers(i)->mutable_source_buffer()->mutable_slice());
        cb(ck->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kCollectiveBroadcastThunk: {
      auto* cb_thunk = thunk->mutable_collective_broadcast_thunk();
      for (int i = 0; i < cb_thunk->buffers_size(); ++i) {
        cb(cb_thunk->mutable_buffers(i)
               ->mutable_source_buffer()
               ->mutable_slice());
        cb(cb_thunk->mutable_buffers(i)
               ->mutable_destination_buffer()
               ->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kSendThunk: {
      auto* st = thunk->mutable_send_thunk();
      cb(st->mutable_buffer()->mutable_source_buffer()->mutable_slice());
      cb(st->mutable_buffer()->mutable_destination_buffer()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kRecvThunk: {
      auto* rt = thunk->mutable_recv_thunk();
      cb(rt->mutable_buffer()->mutable_source_buffer()->mutable_slice());
      cb(rt->mutable_buffer()->mutable_destination_buffer()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kTriangularSolveThunk: {
      auto* ts = thunk->mutable_triangular_solve_thunk();
      cb(ts->mutable_a_buffer()->mutable_slice());
      cb(ts->mutable_b_buffer()->mutable_slice());
      cb(ts->mutable_temp_buffer()->mutable_slice());
      return absl::OkStatus();
    }
    case ThunkProto::kNormThunk: {
      auto* nt = thunk->mutable_norm_thunk();
      cb(nt->mutable_x());
      cb(nt->mutable_scale());
      cb(nt->mutable_y_or_dx());
      if (nt->has_bias()) cb(nt->mutable_bias());
      if (nt->has_expectation()) cb(nt->mutable_expectation());
      if (nt->has_norm_factor()) cb(nt->mutable_norm_factor());
      return absl::OkStatus();
    }
    case ThunkProto::kConvolutionThunk: {
      auto* ct = thunk->mutable_convolution_thunk();
      for (int i = 0; i < ct->operand_buffers_size(); ++i) {
        cb(ct->mutable_operand_buffers(i)->mutable_slice());
      }
      for (int i = 0; i < ct->result_buffers_size(); ++i) {
        cb(ct->mutable_result_buffers(i)->mutable_slice());
      }
      cb(ct->mutable_scratch_buffer());
      return absl::OkStatus();
    }
    case ThunkProto::kConvolutionReorderThunk: {
      auto* crt = thunk->mutable_convolution_reorder_thunk();
      cb(crt->mutable_filter_input()->mutable_slice());
      cb(crt->mutable_filter_output()->mutable_slice());
      if (crt->has_biases()) {
        cb(crt->mutable_biases()->mutable_bias_input()->mutable_slice());
        cb(crt->mutable_biases()->mutable_bias_output()->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kCudnnThunk: {
      auto* cudnn = thunk->mutable_cudnn_thunk();
      for (int i = 0; i < cudnn->args_size(); ++i) {
        cb(cudnn->mutable_args(i)->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kFftThunk: {
      auto* fft = thunk->mutable_fft_thunk();
      cb(fft->mutable_input_buffer());
      cb(fft->mutable_output_buffer());
      return absl::OkStatus();
    }
    case ThunkProto::kSelectKThunk: {
      auto* sk = thunk->mutable_select_k_thunk();
      for (int i = 0; i < sk->args_size(); ++i) {
        cb(sk->mutable_args(i));
      }
      return absl::OkStatus();
    }
    case ThunkProto::kInfeedThunk: {
      auto* infeed = thunk->mutable_infeed_thunk();
      for (int i = 0; i < infeed->dest_slices_size(); ++i) {
        cb(infeed->mutable_dest_slices(i)->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kOutfeedThunk: {
      auto* outfeed = thunk->mutable_outfeed_thunk();
      for (int i = 0; i < outfeed->source_slices_size(); ++i) {
        cb(outfeed->mutable_source_slices(i)->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kHostSendThunk: {
      cb(thunk->mutable_host_send_thunk()->mutable_buffer());
      return absl::OkStatus();
    }
    case ThunkProto::kHostRecvThunk: {
      cb(thunk->mutable_host_recv_thunk()->mutable_buffer());
      return absl::OkStatus();
    }
    case ThunkProto::kHostExecuteStartThunk: {
      auto* hes = thunk->mutable_host_execute_start_thunk();
      for (int i = 0; i < hes->args_size(); ++i) {
        cb(hes->mutable_args(i)->mutable_slice());
      }
      for (int i = 0; i < hes->results_size(); ++i) {
        cb(hes->mutable_results(i)->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kHostExecuteDoneThunk: {
      auto* hed = thunk->mutable_host_execute_done_thunk();
      for (int i = 0; i < hed->results_size(); ++i) {
        cb(hed->mutable_results(i)->mutable_slice());
      }
      return absl::OkStatus();
    }
    case ThunkProto::kRngSeedThunk: {
      cb(thunk->mutable_rng_seed_thunk()->mutable_dest_buffer());
      return absl::OkStatus();
    }
    case ThunkProto::kSequentialThunk: {
      for (auto& child : *thunk->mutable_sequential_thunk()->mutable_thunks()) {
        auto status = ForEachBufferSlice(&child, cb);
        if (!status.ok()) return status;
      }
      return absl::OkStatus();
    }
    case ThunkProto::kCollectiveGroupThunk: {
      for (auto& child :
           *thunk->mutable_collective_group_thunk()->mutable_thunks()) {
        auto status = ForEachBufferSlice(&child, cb);
        if (!status.ok()) return status;
      }
      return absl::OkStatus();
    }
    case ThunkProto::kAsyncStartThunk: {
      for (auto& child : *thunk->mutable_async_start_thunk()
                              ->mutable_thunks()
                              ->mutable_thunks()) {
        auto status = ForEachBufferSlice(&child, cb);
        if (!status.ok()) return status;
      }
      return absl::OkStatus();
    }
    case ThunkProto::kConditionalThunk: {
      auto* cond = thunk->mutable_conditional_thunk();
      cb(cond->mutable_branch_index_buffer()->mutable_slice());
      for (auto& seq : *cond->mutable_branch_thunks()) {
        for (auto& child : *seq.mutable_thunks()) {
          auto status = ForEachBufferSlice(&child, cb);
          if (!status.ok()) return status;
        }
      }
      return absl::OkStatus();
    }
    case ThunkProto::kWhileThunk: {
      auto* wt = thunk->mutable_while_thunk();
      cb(wt->mutable_condition_result_buffer_index());
      for (auto& child :
           *wt->mutable_condition_thunk_sequence()->mutable_thunks()) {
        auto status = ForEachBufferSlice(&child, cb);
        if (!status.ok()) return status;
      }
      for (auto& child : *wt->mutable_body_thunk_sequence()->mutable_thunks()) {
        auto status = ForEachBufferSlice(&child, cb);
        if (!status.ok()) return status;
      }
      return absl::OkStatus();
    }
    case ThunkProto::kReplicaIdThunk:
    case ThunkProto::kPartitionIdThunk:
    case ThunkProto::kAsyncDoneThunk:
    case ThunkProto::kHostSendDoneThunk:
    case ThunkProto::kHostRecvDoneThunk:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported thunk proto kind in call_thunk_proto: ",
                       thunk->impl_case()));
  }
}

}  // namespace xla::gpu

#endif  // JAXLIB_THUNKY_THUNK_PROTO_UTIL_H_
