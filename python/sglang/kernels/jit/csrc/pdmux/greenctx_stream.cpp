#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cuda.h>
#include <numeric>
#include <vector>

namespace sglang {

using GreenCtxStreamCreate = CUresult(CUDAAPI*)(CUstream*, CUgreenCtx, unsigned int, int);

static void check_cuda(CUresult status, const char* call) {
  if (status == CUDA_SUCCESS) return;
  const char* error = nullptr;
  cuGetErrorString(status, &error);
  TVM_FFI_ICHECK(false) << call << " failed: " << (error ? error : "unknown CUDA driver error");
}

static tvm::ffi::Array<int64_t>
create_overlapped_greenctx_stream_by_value(int64_t prefill_reserved_sm, int64_t decode_reserved_sm, int64_t device) {
  int driver_version = 0;
  check_cuda(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");

  CUdevResource input{};
  check_cuda(
      cuDeviceGetDevResource(static_cast<CUdevice>(device), &input, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
  TVM_FFI_ICHECK(
      prefill_reserved_sm > 0 && decode_reserved_sm > 0 && prefill_reserved_sm + decode_reserved_sm < input.sm.smCount)
      << "The lane reservations must be positive and leave shared SMs";

  std::vector<CUdevResource> exclusive;
  CUdevResource shared{};
  unsigned int prefill_group_count = 1;
  unsigned int decode_group_count = 1;
  if (prefill_reserved_sm == decode_reserved_sm) {
    // Preserve the original behavior: CUDA may round both equal floors up to
    // a supported symmetric partition size (for example 28 -> 32 on Hopper).
    exclusive.resize(2);
    unsigned int group_count = 2;
    check_cuda(
        cuDevSmResourceSplitByCount(
            exclusive.data(), &group_count, &input, &shared, 0, static_cast<unsigned int>(prefill_reserved_sm)),
        "cuDevSmResourceSplitByCount");
    TVM_FFI_ICHECK(group_count == 2) << "CUDA could not create two equal protected SM partitions";
  } else {
    // CUDA 13.0 only exposes the equal-group split API. Build unequal lane
    // floors from multiple equal groups produced by one split, so all pieces
    // can legally be combined with the shared remainder in either descriptor.
    const int64_t group_sm = std::gcd(prefill_reserved_sm, decode_reserved_sm);
    const int64_t alignment = input.sm.smCoscheduledAlignment;
    const int64_t min_partition = input.sm.minSmPartitionSize;
    TVM_FFI_ICHECK(group_sm >= min_partition && group_sm % alignment == 0)
        << "Asymmetric PDMux reservations must have a greatest common divisor "
        << "that is at least the CUDA minimum partition size (" << min_partition
        << ") and a multiple of the SM alignment (" << alignment << ")";
    prefill_group_count = static_cast<unsigned int>(prefill_reserved_sm / group_sm);
    decode_group_count = static_cast<unsigned int>(decode_reserved_sm / group_sm);
    unsigned int group_count = prefill_group_count + decode_group_count;
    const unsigned int requested_group_count = group_count;
    exclusive.resize(group_count);
    check_cuda(
        cuDevSmResourceSplitByCount(
            exclusive.data(), &group_count, &input, &shared, 0, static_cast<unsigned int>(group_sm)),
        "cuDevSmResourceSplitByCount");
    TVM_FFI_ICHECK(group_count == requested_group_count)
        << "CUDA created " << group_count << " protected groups, but " << requested_group_count
        << " are required for the asymmetric layout";
    for (const auto& resource : exclusive) {
      TVM_FFI_ICHECK(resource.sm.smCount == group_sm)
          << "CUDA rounded an asymmetric protected group from " << group_sm << " to " << resource.sm.smCount
          << " SMs; choose reservations aligned to the device partition size";
    }
  }
  TVM_FFI_ICHECK(shared.sm.smCount > 0) << "The requested SM reservations leave no shared partition";

  std::vector<CUdevResource> prefill_resources(exclusive.begin(), exclusive.begin() + prefill_group_count);
  prefill_resources.push_back(shared);
  std::vector<CUdevResource> decode_resources(exclusive.begin() + prefill_group_count, exclusive.end());
  decode_resources.push_back(shared);
  CUdevResourceDesc desc[2]{};
  CUgreenCtx green_ctx[2]{};
  check_cuda(
      cuDevResourceGenerateDesc(&desc[0], prefill_resources.data(), prefill_resources.size()),
      "cuDevResourceGenerateDesc(prefill)");
  check_cuda(
      cuDevResourceGenerateDesc(&desc[1], decode_resources.data(), decode_resources.size()),
      "cuDevResourceGenerateDesc(decode)");
  check_cuda(
      cuGreenCtxCreate(&green_ctx[0], desc[0], static_cast<CUdevice>(device), CU_GREEN_CTX_DEFAULT_STREAM),
      "cuGreenCtxCreate(prefill)");
  check_cuda(
      cuGreenCtxCreate(&green_ctx[1], desc[1], static_cast<CUdevice>(device), CU_GREEN_CTX_DEFAULT_STREAM),
      "cuGreenCtxCreate(decode)");

  GreenCtxStreamCreate create_stream = nullptr;
  const CUresult lookup =
      cuGetProcAddress("cuGreenCtxStreamCreate", reinterpret_cast<void**>(&create_stream), driver_version, 0, nullptr);
  CUstream streams[2]{};
  if (lookup == CUDA_SUCCESS && create_stream != nullptr) {
    check_cuda(create_stream(&streams[0], green_ctx[0], CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate(prefill)");
    check_cuda(create_stream(&streams[1], green_ctx[1], CU_STREAM_NON_BLOCKING, -1), "cuGreenCtxStreamCreate(decode)");
  } else {
    CUcontext context{};
    check_cuda(cuCtxFromGreenCtx(&context, green_ctx[0]), "cuCtxFromGreenCtx(prefill)");
    check_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent(prefill)");
    check_cuda(cuStreamCreate(&streams[0], CU_STREAM_NON_BLOCKING), "cuStreamCreate(prefill)");
    check_cuda(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent(prefill)");

    check_cuda(cuCtxFromGreenCtx(&context, green_ctx[1]), "cuCtxFromGreenCtx(decode)");
    check_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent(decode)");
    check_cuda(
        cuStreamCreateWithPriority(&streams[1], CU_STREAM_NON_BLOCKING, -1), "cuStreamCreateWithPriority(decode)");
    check_cuda(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent(decode)");
  }

  tvm::ffi::Array<int64_t> result;
  result.reserve(7);
  result.push_back(static_cast<int64_t>(reinterpret_cast<intptr_t>(streams[0])));
  result.push_back(static_cast<int64_t>(reinterpret_cast<intptr_t>(streams[1])));
  int64_t actual_prefill_only = 0;
  int64_t actual_decode_only = 0;
  for (unsigned int i = 0; i < prefill_group_count; ++i) {
    actual_prefill_only += exclusive[i].sm.smCount;
  }
  for (unsigned int i = prefill_group_count; i < exclusive.size(); ++i) {
    actual_decode_only += exclusive[i].sm.smCount;
  }
  result.push_back(actual_prefill_only + shared.sm.smCount);
  result.push_back(actual_decode_only + shared.sm.smCount);
  result.push_back(actual_prefill_only);
  result.push_back(actual_decode_only);
  result.push_back(static_cast<int64_t>(shared.sm.smCount));
  return result;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_overlapped_greenctx_stream_by_value, create_overlapped_greenctx_stream_by_value);

}  // namespace sglang
