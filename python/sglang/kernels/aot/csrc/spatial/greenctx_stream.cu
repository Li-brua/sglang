// Documentation: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
#include <torch/all.h>

#include <cstdlib>
#include <numeric>
#include <vector>

#include "cuda_utils.h"
#include "greenctx_stream.h"

static int CUDA_DRIVER_VERSION;

using PFN_cuGreenCtxStreamCreate = CUresult(CUDAAPI*)(CUstream*, CUgreenCtx, unsigned int, int);

auto probe_cuGreenCtxStreamCreate() -> PFN_cuGreenCtxStreamCreate {
  static PFN_cuGreenCtxStreamCreate pfn = nullptr;
  CUDA_DRV(cuGetProcAddress("cuGreenCtxStreamCreate", reinterpret_cast<void**>(&pfn), CUDA_DRIVER_VERSION, 0, nullptr));
  return pfn;
}

static std::vector<int64_t> create_greenctx_stream_fallback(CUgreenCtx gctx[2], int decode_priority = 0) {
  CUstream streamA, streamB;
  CUcontext ctx;

  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[0]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamA, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[1]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreateWithPriority(&streamB, CU_STREAM_NON_BLOCKING, decode_priority));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  return {(int64_t)streamA, (int64_t)streamB};
}

inline void destroy_green_context(CUgreenCtx gctx) {
  if (!gctx) return;
  CUDA_DRV(cuGreenCtxDestroy(gctx));
}

static std::vector<int64_t> create_greenctx_stream_direct_dynamic(CUgreenCtx gctx[2], int decode_priority = 0) {
  // This symbol is introduced in CUDA 12.5
  const static auto pfn = probe_cuGreenCtxStreamCreate();
  if (!pfn) {
    TORCH_WARN("cuGreenCtxStreamCreate(cuda>=12.5) is not available, using fallback");
    return create_greenctx_stream_fallback(gctx, decode_priority);
  }

  CUstream streamA, streamB;
  CUDA_DRV(pfn(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(pfn(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, decode_priority));

  return {(int64_t)streamA, (int64_t)streamB};
}

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  CUDA_DRV(cuDriverGetVersion(&CUDA_DRIVER_VERSION));

  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];

  TORCH_CHECK(smA > 0 && smB > 0, "SM counts must be positive");

  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));

  const unsigned minCount = static_cast<unsigned>(smA + smB);
  const unsigned minCountA = static_cast<unsigned>(smA);
  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  unsigned nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  const int smCountA = resources[0].sm.smCount;
  const int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> streams = create_greenctx_stream_direct_dynamic(gctx);

  destroy_green_context(gctx[2]);

  std::vector<int64_t> vec = {
      streams[0],  // streamA
      streams[1],  // streamB
      (int64_t)smCountA,
      (int64_t)smCountB};

  return vec;
}

static std::vector<int64_t>
create_protected_overlapped_greenctx_stream(int64_t prefill_reserved_sm, int64_t decode_reserved_sm, int64_t device) {
  CUDA_DRV(cuDriverGetVersion(&CUDA_DRIVER_VERSION));

  CUdevResource input;
  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));
  TORCH_CHECK(
      prefill_reserved_sm > 0 && decode_reserved_sm > 0 && prefill_reserved_sm + decode_reserved_sm < input.sm.smCount,
      "The lane reservations must be positive and leave shared SMs");

  // One split produces all disjoint reservations plus the shared remainder.
  // CUDA requires every resource combined in a descriptor to come from the
  // same split operation.
  std::vector<CUdevResource> exclusive;
  CUdevResource shared;
  unsigned prefill_group_count = 1;
  unsigned decode_group_count = 1;
  if (prefill_reserved_sm == decode_reserved_sm) {
    // Keep the legacy symmetric semantics, including CUDA rounding 28 to 32.
    exclusive.resize(2);
    unsigned nbGroups = 2;
    CUDA_DRV(cuDevSmResourceSplitByCount(
        exclusive.data(), &nbGroups, &input, &shared, 0, static_cast<unsigned>(prefill_reserved_sm)));
    TORCH_CHECK(nbGroups == 2, "CUDA could not create two equal protected SM partitions");
  } else {
    // CUDA 13.0 has only the equal-group split API. Unequal floors are made
    // from multiple equal groups from one split. For 8/24 this creates four
    // 8-SM groups: one for prefill and three for decode.
    const int64_t group_sm = std::gcd(prefill_reserved_sm, decode_reserved_sm);
    const int64_t alignment = input.sm.smCoscheduledAlignment;
    const int64_t min_partition = input.sm.minSmPartitionSize;
    TORCH_CHECK(
        group_sm >= min_partition && group_sm % alignment == 0,
        "Asymmetric PDMux reservations must have a greatest common divisor "
        "that is at least the CUDA minimum partition size (",
        min_partition,
        ") and a multiple of the SM alignment (",
        alignment,
        ")");
    prefill_group_count = static_cast<unsigned>(prefill_reserved_sm / group_sm);
    decode_group_count = static_cast<unsigned>(decode_reserved_sm / group_sm);
    unsigned nbGroups = prefill_group_count + decode_group_count;
    const unsigned requested_groups = nbGroups;
    exclusive.resize(nbGroups);
    CUDA_DRV(
        cuDevSmResourceSplitByCount(exclusive.data(), &nbGroups, &input, &shared, 0, static_cast<unsigned>(group_sm)));
    TORCH_CHECK(
        nbGroups == requested_groups,
        "CUDA created ",
        nbGroups,
        " protected groups, but ",
        requested_groups,
        " are required for the asymmetric layout");
    for (const auto& resource : exclusive) {
      TORCH_CHECK(
          resource.sm.smCount == group_sm,
          "CUDA rounded an asymmetric protected group from ",
          group_sm,
          " to ",
          resource.sm.smCount,
          " SMs; choose reservations aligned to the device partition size");
    }
  }
  TORCH_CHECK(shared.sm.smCount > 0, "The requested SM reservations leave no shared partition");

  std::vector<CUdevResource> prefill_resources(exclusive.begin(), exclusive.begin() + prefill_group_count);
  prefill_resources.push_back(shared);
  std::vector<CUdevResource> decode_resources(exclusive.begin() + prefill_group_count, exclusive.end());
  decode_resources.push_back(shared);
  CUdevResourceDesc desc[2];
  CUgreenCtx gctx[2];
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], prefill_resources.data(), prefill_resources.size()));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], decode_resources.data(), decode_resources.size()));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  // Decode retains its overlay priority, but its stream cannot reach the
  // prefill-only reservation. Report the driver's actual, possibly rounded SM
  // counts rather than the requested minimum.
  auto streams = create_greenctx_stream_direct_dynamic(gctx, -1);
  int64_t actual_prefill_only = 0;
  int64_t actual_decode_only = 0;
  for (unsigned i = 0; i < prefill_group_count; ++i) {
    actual_prefill_only += exclusive[i].sm.smCount;
  }
  for (unsigned i = prefill_group_count; i < exclusive.size(); ++i) {
    actual_decode_only += exclusive[i].sm.smCount;
  }
  return {
      streams[0],
      streams[1],
      actual_prefill_only + shared.sm.smCount,
      actual_decode_only + shared.sm.smCount,
      actual_prefill_only,
      actual_decode_only,
      static_cast<int64_t>(shared.sm.smCount)};
}

std::vector<int64_t> create_overlapped_greenctx_stream_by_value(int64_t reserved_sm, int64_t device) {
  return create_protected_overlapped_greenctx_stream(reserved_sm, reserved_sm, device);
}

std::vector<int64_t> create_asymmetric_overlapped_greenctx_stream_by_value(
    int64_t prefill_reserved_sm, int64_t decode_reserved_sm, int64_t device) {
  return create_protected_overlapped_greenctx_stream(prefill_reserved_sm, decode_reserved_sm, device);
}
