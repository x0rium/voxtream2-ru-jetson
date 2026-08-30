#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Fuse the operation between two dep_former TensorRT enqueues:
//   BF16 conditional/unconditional logits -> CFG argmax -> exact BF16 row copy.
// Argmax(softmax(logits / temperature)) is exactly argmax(logits), so omitting
// softmax does not alter the selected acoustic token.
extern "C" __global__ void acoustic_cfg_argmax_embed(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* embedding_weight,
    int64_t* frame_codes,
    int codebook,
    float cfg_gamma,
    __nv_bfloat16* next_hidden) {
  __shared__ float best_values[256];
  __shared__ int best_indices[256];

  const int tid = threadIdx.x;
  float best_value = -3.402823466e+38F;
  int best_index = 0;
  for (int index = tid; index < 2050; index += blockDim.x) {
    const float conditional = __bfloat162float(logits[index]);
    const float unconditional = __bfloat162float(logits[2050 + index]);
    const float value =
        cfg_gamma * conditional + (1.0f - cfg_gamma) * unconditional;
    if (value > best_value || (value == best_value && index < best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  best_values[tid] = best_value;
  best_indices[tid] = best_index;
  __syncthreads();

  for (int width = blockDim.x / 2; width; width >>= 1) {
    if (tid < width) {
      const float other_value = best_values[tid + width];
      const int other_index = best_indices[tid + width];
      if (other_value > best_values[tid] ||
          (other_value == best_values[tid] &&
           other_index < best_indices[tid])) {
        best_values[tid] = other_value;
        best_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }

  const int token = best_indices[0];
  if (tid == 0) {
    frame_codes[codebook] = static_cast<int64_t>(token);
  }
  __syncthreads();

  const int64_t row = static_cast<int64_t>(codebook) * 2050 + token;
  for (int dimension = tid; dimension < 1024; dimension += blockDim.x) {
    const __nv_bfloat16 word = embedding_weight[row * 1024 + dimension];
    next_hidden[dimension] = word;
    next_hidden[1024 + dimension] = word;
  }
}
