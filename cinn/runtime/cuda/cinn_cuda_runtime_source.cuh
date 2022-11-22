/**
 * \file This file contains all the intrinsics available to be used in CUDA code generated by CodeGen.
 */
extern "C" {
// *************************************************************** //
// float32 unary and binary operator
#define FN_FP32(func) cinn_nvgpu_##func##_fp32
// NOTE Due to function override, we don't need to use type (such as '_fp32') as the suffix of function's name.
__device__ inline float FN_FP32(sin)(float x) { return sin(x); }
__device__ inline float FN_FP32(cos)(float x) { return cos(x); }
__device__ inline float FN_FP32(cosh)(float x) { return cosh(x); }
__device__ inline float FN_FP32(tanh)(float x) { return tanh(x); }

__device__ inline float FN_FP32(asin)(float x) { return asin(x); }
__device__ inline float FN_FP32(acos)(float x) { return acos(x); }
__device__ inline float FN_FP32(acosh)(float x) { return acosh(x); }
__device__ inline float FN_FP32(atanh)(float x) { return atanh(x); }

__device__ inline float FN_FP32(ceil)(float x) { return ceil(x); }
__device__ inline float FN_FP32(round)(float x) { return round(x); }
__device__ inline float FN_FP32(trunc)(float x) { return trunc(x); }
__device__ inline float FN_FP32(abs)(float x) { return abs(x); }
__device__ inline float FN_FP32(floor)(float x) { return floor(x); }
__device__ inline float FN_FP32(log)(float x) { return log(x); }
__device__ inline float FN_FP32(log2)(float x) { return log2(x); }
__device__ inline float FN_FP32(log10)(float x) { return log10(x); }
__device__ inline float FN_FP32(exp)(float x) { return exp(x); }
__device__ inline float FN_FP32(erf)(float x) { return erf(x); }
__device__ inline float FN_FP32(sigmoid)(float x) { return 1. / (1 + exp(-x)); }
__device__ inline float FN_FP32(sqrt)(float x) { return sqrt(x); }
__device__ inline float FN_FP32(rsqrt)(float x) { return rsqrt(x); }

__device__ inline bool FN_FP32(isfinite)(float x) { return isfinite(x); }
__device__ inline bool FN_FP32(isinf)(float x) { return isinf(x); }
__device__ inline bool FN_FP32(isnan)(float x) { return isnan(x); }

__device__ inline float FN_FP32(max)(float a, float b) { return max(a, b); }
__device__ inline float FN_FP32(min)(float a, float b) { return min(a, b); }

__device__ inline float FN_FP32(pow)(float a, float b) { return powf(a, b); }

__device__ inline float FN_FP32(remainder)(float a, float b) { return remainderf(a, b); }


// *************************************************************** //
// float64 unary and binary operator
#define FN_FP64(func) cinn_nvgpu_##func##_fp64

__device__ inline double FN_FP64(pow)(double a, double b) { return pow(a, b); }


// *************************************************************** //
// int32 unary and binary operator
#define FN_INT32(func) cinn_nvgpu_##func##_int32

__device__ inline int FN_INT32(pow)(int a, int b) {
  int res = 1;
  for (int i = 0; i < b; ++i) {
    res *= a;
  }
  return res;
}

__device__ inline int FN_INT32(left_shift)(int a, int b) { return a << b; }
__device__ inline int FN_INT32(right_shift)(int a, int b) { return a >> b; }
__device__ inline int FN_INT32(bitwise_and)(int a, int b) { return a & b; }
__device__ inline int FN_INT32(bitwise_or)(int a, int b) { return a | b; }
__device__ inline int FN_INT32(bitwise_xor)(int a, int b) { return a ^ b; }
__device__ inline int FN_INT32(bitwise_not)(int a) { return ~a; }
__device__ inline int FN_INT32(clz)(int a) { return __clz(a); }
__device__ inline int FN_INT32(popc)(int a) { return __popc(a); }


// *************************************************************** //
// int64 unary and binary operator
#define FN_INT64(func) cinn_nvgpu_##func##_int64

__device__ inline long long int FN_INT64(clz)(long long int a) { return __clzll(a); }
__device__ inline long long int FN_INT64(popc)(long long int a) { return __popcll(a); }


// *************************************************************** //
// float16 unary and binary operator
#ifdef CINN_CUDA_FP16

#define FN_FP16(func) cinn_nvgpu_##func##_fp16

__device__ inline float16 FN_FP16(ceil)(float16 x) { return float16(hceil(x.to_half())); }
__device__ inline float16 FN_FP16(floor)(float16 x) { return float16(hfloor(x.to_half())); }
__device__ inline float16 FN_FP16(round)(float16 x) { return float16(hrint(x.to_half())); }
__device__ inline float16 FN_FP16(trunc)(float16 x) { return float16(htrunc(x.to_half())); }

__device__ inline float16 FN_FP16(sin)(float16 x) { return float16(hsin(x.to_half())); }
__device__ inline float16 FN_FP16(cos)(float16 x) { return float16(hcos(x.to_half())); }

__device__ inline float16 FN_FP16(exp)(float16 x) { return float16(hexp(x.to_half())); }
__device__ inline float16 FN_FP16(log)(float16 x) { return float16(hlog(x.to_half())); }
__device__ inline float16 FN_FP16(log2)(float16 x) { return float16(hlog2(x.to_half())); }
__device__ inline float16 FN_FP16(log10)(float16 x) { return float16(hlog10(x.to_half())); }

__device__ inline float16 FN_FP16(sqrt)(float16 x) { return float16(hsqrt(x.to_half())); }
__device__ inline float16 FN_FP16(rsqrt)(float16 x) { return float16(hrsqrt(x.to_half())); }

__device__ inline float16 FN_FP16(abs)(float16 x) { return cinn::common::abs(x); }

__device__ inline bool FN_FP16(isnan)(float16 x) { return cinn::common::isnan(x); }
__device__ inline bool FN_FP16(isinf)(float16 x) { return cinn::common::isinf(x); }
__device__ inline bool FN_FP16(isfinite)(float16 x) { return cinn::common::isfinite(x); }

__device__ inline float16 FN_FP16(erf)(float16 x) { return float16(FN_FP32(erf)(static_cast<float>(x))); }

__device__ inline float16 FN_FP16(cosh)(float16 x) { return float16(FN_FP32(cosh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(tanh)(float16 x) { return float16(FN_FP32(tanh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(asin)(float16 x) { return float16(FN_FP32(asin)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(acos)(float16 x) { return float16(FN_FP32(acos)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(acosh)(float16 x) { return float16(FN_FP32(acosh)(static_cast<float>(x))); }
__device__ inline float16 FN_FP16(atanh)(float16 x) { return float16(FN_FP32(atanh)(static_cast<float>(x))); }

__device__ inline float16 FN_FP16(sigmoid)(float16 x) { return float16(FN_FP32(sigmoid)(static_cast<float>(x))); }

__device__ inline float16 FN_FP16(max)(float16 a, float16 b) { return a > b ? a : b; }
__device__ inline float16 FN_FP16(min)(float16 a, float16 b) { return a < b ? a : b; }

__device__ inline float16 FN_FP16(remainder)(float16 a, float16 b) { return float16(FN_FP32(remainder)(static_cast<float>(a), static_cast<float>(b))); }
__device__ inline float16 FN_FP16(pow)(float16 a, float16 b) { return float16(FN_FP32(pow)(static_cast<float>(a), static_cast<float>(b))); }

#endif


// *************************************************************** //
// reduce operator, need `--expt-relaxed-constexpr` option to call std function in device kernel
#define EXPAND_REDUCE_FP32_MACRO(MACRO, ...) \
  MACRO(sum_fp32, 0.0f, float, ##__VA_ARGS__) \
  MACRO(prod_fp32, 1.0f, float, ##__VA_ARGS__) \
  MACRO(max_fp32, -3.40282e+38, float, ##__VA_ARGS__) \
  MACRO(min_fp32, 3.40282e+38, float, ##__VA_ARGS__) \

__device__ inline float cinn_sum_fp32(const float left, const float right) { return left + right; }
__device__ inline float cinn_prod_fp32(const float left, const float right) { return left * right; }
__device__ inline float cinn_max_fp32(const float left, const float right) { return FN_FP32(max)(left, right); }
__device__ inline float cinn_min_fp32(const float left, const float right) { return FN_FP32(min)(left, right); }


#ifdef CINN_CUDA_FP16

#define EXPAND_REDUCE_FP16_MACRO(MACRO, ...) \
  MACRO(sum_fp16, float16(0.0), float16, ##__VA_ARGS__) \
  MACRO(prod_fp16, float16(1.0), float16, ##__VA_ARGS__) \
  MACRO(max_fp16, cinn::common::raw_uint16_to_float16(0xfbff), float16, ##__VA_ARGS__) \
  MACRO(min_fp16, cinn::common::raw_uint16_to_float16(0x7bff), float16, ##__VA_ARGS__)

__device__ inline float16 cinn_sum_fp16(const float16 left, const float16 right) { return left + right; }
__device__ inline float16 cinn_prod_fp16(const float16 left, const float16 right) { return left * right; }
__device__ inline float16 cinn_max_fp16(const float16 left, const float16 right) { return FN_FP16(max)(left, right); }
__device__ inline float16 cinn_min_fp16(const float16 left, const float16 right) { return FN_FP16(min)(left, right); }
#endif

#define EXPAND_REDUCE_BOOL_MACRO(MACRO, ...) \
  MACRO(all, true, bool, ##__VA_ARGS__) \
  MACRO(any, false, bool, ##__VA_ARGS__)

__device__ inline bool cinn_all(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any(const bool left, const bool right) { return left || right; }


#define CINN_SHUFFLE_FUNCTION(offset, op, init)                  \
  shfl_res = __shfl_down_sync(mask, tmp_val, offset, 32);        \
  shfl_res = threadIdx.x % 32 + offset < lane ? shfl_res : init; \
  tmp_val  = op(tmp_val, shfl_res);

#define CINN_WARP_SHUFFLE_INTERNAL_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_warp_shuffle_##REDUCE_TYPE##_internal(const DTYPE value) { \
  DTYPE tmp_val      = value, shfl_res; \
  unsigned int mask = __activemask(); \
  unsigned int lane = __popc(mask); \
  CINN_SHUFFLE_FUNCTION(16, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(8, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(4, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(2, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(1, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  tmp_val = __shfl_sync(mask, tmp_val, 0, 32); \
  return tmp_val; \
}

EXPAND_REDUCE_FP32_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
#endif

#undef CINN_WARP_SHUFFLE_INTERNAL_IMPL

#define CINN_WARP_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_warp_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
  DTYPE tmp_val = DTYPE(INITIAL_VALUE); \
  for (int i = threadIdx.x; i < extend; i += 32) { \
    tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]); \
  } \
  return cinn_warp_shuffle_##REDUCE_TYPE##_internal(tmp_val); \
}

EXPAND_REDUCE_FP32_MACRO(CINN_WARP_REDUCE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_REDUCE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_REDUCE_IMPL)
#endif

#undef CINN_WARP_REDUCE_IMPL

__device__ inline float cinn_warp_reduce_avg_fp32(const float *buf, int offset, int extend) {
  return cinn_warp_reduce_sum_fp32(buf, offset, extend) / extend;
}

#define CINN_BLOCK_REDUCE_INTERNAL_IMPL(TYPE, value, init_value, cinn_warp_shuffle_internal) \
  int warp_id = threadIdx.x / 32;                                                              \
  __shared__ TYPE tmp[32];                                                                     \
  if (warp_id == 0) {                                                                          \
    tmp[threadIdx.x] = init_value;                                                             \
  }                                                                                            \
  TYPE tmp_val = cinn_warp_shuffle_internal(value);                                            \
  if (blockDim.x <= 32) {                                                                      \
    return tmp_val;                                                                            \
  }                                                                                            \
  __syncthreads();                                                                             \
  if (threadIdx.x % 32 == 0) {                                                                 \
    tmp[warp_id] = tmp_val;                                                                    \
  }                                                                                            \
  __syncthreads();                                                                             \
  if (warp_id == 0) {                                                                          \
    tmp_val = tmp[threadIdx.x];                                                                \
    tmp_val = cinn_warp_shuffle_internal(tmp_val);                                             \
    if (threadIdx.x == 0) {                                                                    \
      tmp[0] = tmp_val;                                                                        \
    }                                                                                          \
  }                                                                                            \
  __syncthreads();                                                                             \
  return tmp[0];

#define CINN_BLOCK_REDUCE_INTERNAL_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE##_internal(const DTYPE value) { \
  CINN_BLOCK_REDUCE_INTERNAL_IMPL(DTYPE, value, DTYPE(INITIAL_VALUE), cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
}

EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
#endif

#undef CINN_BLOCK_REDUCE_INTERNAL_IMPL
#undef CINN_BLOCK_REDUCE_INTERNAL_MACRO

#define CINN_BLOCK_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
  DTYPE tmp_val = DTYPE(INITIAL_VALUE); \
  for (int i = threadIdx.x; i < extend; i += blockDim.x) { \
    tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]); \
  } \
  return cinn_block_reduce_##REDUCE_TYPE##_internal(tmp_val); \
}

EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_IMPL)
#endif

#undef CINN_BLOCK_REDUCE_IMPL

#define BLOCK_SHUFFLE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE block_shuffle_##REDUCE_TYPE(const DTYPE *buf, int line, int stride) { \
  DTYPE val = DTYPE(INITIAL_VALUE); \
  for (int idx = threadIdx.x; idx < line; idx += stride) { \
    val = cinn_##REDUCE_TYPE(val, buf[idx]); \
  } \
  return val; \
}

EXPAND_REDUCE_FP32_MACRO(BLOCK_SHUFFLE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(BLOCK_SHUFFLE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(BLOCK_SHUFFLE_IMPL)
#endif

#undef BLOCK_SHUFFLE_IMPL

#undef EXPAND_REDUCE_FP32_MACRO
#undef EXPAND_REDUCE_BOOL_MACRO

#ifdef CINN_CUDA_FP16
#undef EXPAND_REDUCE_FP16_MACRO
#endif


// *************************************************************** //
// other function
#define __cinn_cuda_find_kernel(buf, size, num, begin, stride)           \
  do {                                                                   \
    for (int i = (size - 1) * stride + begin; i >= begin; i -= stride) { \
      if (buf[i] == num) return (i - begin) / stride;                    \
    }                                                                    \
    return -1;                                                           \
  } while (0)

__device__ inline int cinn_cuda_find_int(const int *buf, int size, int num) {
  __cinn_cuda_find_kernel(buf, size, num, 0, 1);
}

__device__ inline int cinn_cuda_find_float(const float *buf, int size, float num) {
  __cinn_cuda_find_kernel(buf, size, num, 0, 1);
}

__device__ inline int cinn_cuda_find_int_nd(const int *buf, int size, int num, int begin, int stride) {
  __cinn_cuda_find_kernel(buf, size, num, begin, stride);
}

__device__ inline int cinn_cuda_find_float_nd(const float *buf, int size, float num, int begin, int stride) {
  __cinn_cuda_find_kernel(buf, size, num, begin, stride);
}

#undef __cinn_cuda_find_kernel

#define __cinn_cuda_find_from_kernel(buf, size, num, begin) \
  do {                                                      \
    for (int i = begin; i < size; ++i) {                    \
      if (buf[i] == num) return i;                          \
    }                                                       \
    return -1;                                              \
  } while (0)

__device__ inline int cinn_cuda_find_int_from(const int *buf, int size, int num, int begin) {
  __cinn_cuda_find_from_kernel(buf, size, num, begin);
}

__device__ inline int cinn_cuda_find_float_from(const float *buf, int size, float num, int begin) {
  __cinn_cuda_find_from_kernel(buf, size, num, begin);
}

#undef __cinn_cuda_find_from_kernel

#define __cinn_cuda_lt_num_kernel(buf, size, num, offset, stride)          \
  do {                                                                     \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (buf[i] < num) out++;                                             \
    }                                                                      \
    return out;                                                            \
  } while (0)

__device__ inline int cinn_cuda_lt_num_float(
    const float *buf, const int size, const float num, const int offset, const int stride) {
  __cinn_cuda_lt_num_kernel(buf, size, num, offset, stride);
}

__device__ inline int cinn_cuda_lt_num_int(
    const int *buf, const int size, const int num, const int offset, const int stride) {
  __cinn_cuda_lt_num_kernel(buf, size, num, offset, stride);
}

#undef __cinn_cuda_lt_num_kernel

#define __cinn_cuda_gt_num_kernel(buf, size, num, offset, stride)          \
  do {                                                                     \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (buf[i] > num) out++;                                             \
    }                                                                      \
    return out;                                                            \
  } while (0)

__device__ inline int cinn_cuda_gt_num_float(
    const float *buf, const int size, const float num, const int offset, const int stride) {
  __cinn_cuda_gt_num_kernel(buf, size, num, offset, stride);
}

__device__ inline int cinn_cuda_gt_num_int(
    const int *buf, const int size, const int num, const int offset, const int stride) {
  __cinn_cuda_gt_num_kernel(buf, size, num, offset, stride);
}

#undef __cinn_cuda_gt_num_kernel

__device__ inline float cinn_cuda_index_add(const float x,
                                            const int axis_indice,
                                            const float *__restrict__ y,
                                            const int offset,
                                            const int stride,
                                            const int *__restrict__ index,
                                            const int index_size) {
  float res = x;
  int idx   = -1;
  do {
    idx = cinn_cuda_find_int_from(index, index_size, axis_indice, idx + 1);
    if (idx >= 0) {
      res += y[offset + idx * stride];
    }
  } while (idx != -1);
  return res;
}


// *************************************************************** //
// end of macro undef
#undef FN_FP32
#undef FN_FP64
#undef FN_INT32
#undef FN_INT64

#ifdef CINN_CUDA_FP16
#undef FN_FP16
#endif
}
