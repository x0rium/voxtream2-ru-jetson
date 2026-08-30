#include <cstdint>

// The embedding table and output contain raw BF16 words.  No arithmetic is
// required here: copying the 16-bit payload preserves PyTorch embedding lookup
// bit-for-bit while keeping the table outside PyTorch.
extern "C" __global__ void gather_bf16_words(
    const std::uint16_t* __restrict__ weights,
    const std::int64_t* __restrict__ indices,
    std::uint16_t* __restrict__ output,
    std::int64_t index_count,
    std::int32_t embedding_dim) {
  const std::int64_t linear =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t element_count = index_count * embedding_dim;
  if (linear >= element_count) {
    return;
  }

  const std::int64_t index_offset = linear / embedding_dim;
  const std::int32_t column = static_cast<std::int32_t>(linear % embedding_dim);
  const std::int64_t row = indices[index_offset];
  output[linear] = weights[row * embedding_dim + column];
}
