#pragma once

#include <cuda/std/cstdint>

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T>
struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template <typename T>
inline __device__ float sum(T v);

template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template <typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

// Define custom FP32 vector data types.
struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

// FP16 vector types for Q, K, V.
template <>
struct Vec<uint16_t, 1> {
  using Type = uint16_t;
};
template <>
struct Vec<uint16_t, 2> {
  using Type = uint32_t;
};
template <>
struct Vec<uint16_t, 4> {
  using Type = uint2;
};
template <>
struct Vec<uint16_t, 8> {
  using Type = uint4;
};

// FP32 accumulator vector types corresponding to Vec.
template <>
struct FloatVec<uint16_t> {
  using Type = float;
};
template <>
struct FloatVec<uint32_t> {
  using Type = float2;
};
template <>
struct FloatVec<uint2> {
  using Type = Float4_;
};
template <>
struct FloatVec<uint4> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline __device__ uint32_t h0_h0(uint16_t a) {
#ifndef USE_ROCM
  uint32_t b;
  asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
  return b;
#else
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u16[0] = a;
  tmp.u16[1] = a;
  return tmp.u32;
#endif
}

inline __device__ float half_to_float(uint16_t h) {
  float f;
#ifndef USE_ROCM
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#else
  asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
#endif
  return f;
}

inline __device__ float2 half2_to_float2(uint32_t v) {
#ifndef USE_ROCM
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
  return make_float2(half_to_float(lo), half_to_float(hi));
#else
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u32 = v;
  float2 ret;
  ret.x = half_to_float(tmp.u16[0]);
  ret.y = half_to_float(tmp.u16[1]);
  return ret;
#endif
}

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

inline __device__ uint32_t float2_to_half2(float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
#else
  tmp.u16[0] = float_to_half(f.x);
  tmp.u16[1] = float_to_half(f.y);
#endif
  return tmp.u32;
}

// Vector addition.
inline __device__ uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint2 add(uint2 a, uint2 b) {
  uint2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ uint4 add(uint4 a, uint4 b) {
  uint4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

// inline __device__ float2 add(uint32_t a, float2 fb) {
//   float2 fa = half2_to_float2(a);
//   return add(fa, fb);
// }

// inline __device__ Float4_ add(uint2 a, Float4_ fb) {
//   Float4_ fc;
//   fc.x = add(a.x, fb.x);
//   fc.y = add(a.y, fb.y);
//   return fc;
// }

// inline __device__ Float8_ add(uint4 a, Float8_ fb) {
//   Float8_ fc;
//   fc.x = add(a.x, fb.x);
//   fc.y = add(a.y, fb.y);
//   fc.z = add(a.z, fb.z);
//   fc.w = add(a.w, fb.w);
//   return fc;
// }

// Vector multiplication.
template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template <>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

template <>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

template <>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

template <>
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

template <>
inline __device__ uint2 mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  return c;
}

template <>
inline __device__ uint4 mul(uint4 a, uint4 b) {
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
  return c;
}

template <>
inline __device__ uint4 mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
  return c;
}

template <>
inline __device__ float mul(uint16_t a, uint16_t b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

template <>
inline __device__ float2 mul(uint32_t a, uint32_t b) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float2 mul(uint16_t a, uint32_t b) {
  return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ Float4_ mul(uint2 a, uint2 b) {
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  return fc;
}

template <>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint4 a, uint4 b) {
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
#ifndef USE_ROCM
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
#else
  asm volatile("v_pk_fma_f16 %0, %1, %2, %3;\n" : "=v"(d) : "v"(a), "v"(b), "v"(c));
#endif
  return d;
}

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) { return fma(h0_h0(a), b, c); }

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
  uint32_t s = h0_h0(a);
  uint2 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
  uint4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
  uint32_t s = h0_h0(a);
  uint4 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

// inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
//   float2 fa = half2_to_float2(a);
//   float2 fb = half2_to_float2(b);
//   return fma(fa, fb, fc);
// }

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) { return fma(h0_h0(a), b, fc); }

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
template <>
inline __device__ float sum(uint16_t v) {
  return half_to_float(v);
}

template <>
inline __device__ float sum(uint32_t v) {
  float2 tmp = half2_to_float2(v);
  return tmp.x + tmp.y;
}

template <>
inline __device__ float sum(uint2 v) {
  uint32_t c = add(v.x, v.y);
  return sum(c);
}

template <>
inline __device__ float sum(uint4 v) {
  uint32_t c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
  return sum(c);
}

template <>
inline __device__ float sum(float v) {
  return v;
}

template <>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template <>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template <>
inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template <>
inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

// From float32 to float16.
inline __device__ void from_float(uint16_t& dst, float src) { dst = float_to_half(src); }

inline __device__ void from_float(uint32_t& dst, float2 src) { dst = float2_to_half2(src); }

inline __device__ void from_float(uint2& dst, Float4_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
}

inline __device__ void from_float(uint4& dst, Float8_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}

// From float16 to float32.
inline __device__ float to_float(uint16_t u) { return half_to_float(u); }

inline __device__ float2 to_float(uint32_t u) { return half2_to_float2(u); }

inline __device__ Float4_ to_float(uint2 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  return tmp;
}

inline __device__ Float8_ to_float(uint4 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  tmp.z = half2_to_float2(u.z);
  tmp.w = half2_to_float2(u.w);
  return tmp;
}

// Zero-out a variable.
inline __device__ void zero(uint16_t& dst) { dst = uint16_t(0); }

// Q*K^T operation.
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

template <int NUM_WARPS, int WARP_SIZE>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}
