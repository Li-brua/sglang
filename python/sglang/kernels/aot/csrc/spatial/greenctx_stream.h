#include <vector>
std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device);
std::vector<int64_t> create_overlapped_greenctx_stream_by_value(int64_t reserved_sm, int64_t device);
std::vector<int64_t> create_asymmetric_overlapped_greenctx_stream_by_value(
    int64_t prefill_reserved_sm, int64_t decode_reserved_sm, int64_t device);
