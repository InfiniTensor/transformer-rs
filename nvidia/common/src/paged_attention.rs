use crate::PtxWapper;
use cuda::{bindings::CUdeviceptr, ContextSpore, CudaDataType, DevByte, ModuleSpore, Ptx, Stream};
use std::{
    ffi::{c_void, CString},
    ops::{Deref, DerefMut},
};
use tensor::{Tensor};

pub struct PagedAttention {
  ptx: Ptx,
  f: CString,
  num_threads: u32,
  warp_size: u32,
}

impl PtxWapper for PagedAttention {
  #[inline]
  fn ptx(&self) -> &Ptx {
      &self.ptx
  }
}

impl PagedAttention {
    pub fn new(
        head_size: u32,
        kv_block_size: u32,
        num_threads: u32,
        warp_size: u32,
        scalar_t: CudaDataType,
        cache_t: CudaDataType,
    ) -> Self {
      let name = "paged_attention";
        let scalar_t = scalar_t.name();
        let cache_t = cache_t.name();

        const PAGED_ATTENTION: &str = include_str!("paged_attention.cuh");
        let code = format!(
            r#"{PAGED_ATTENTION}

            #define MAX(a, b) ((a) > (b) ? (a) : (b))
            #define MIN(a, b) ((a) < (b) ? (a) : (b))
            #define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))
            
            __device__ void {name}(
              {scalar_t}* __restrict__ out, const {scalar_t}* __restrict__ q,
              const {scalar_t}* __restrict__ k_cache, const {scalar_t}* __restrict__ v_cache,
              const int num_kv_heads, const float scale, const int* __restrict__ block_tables,
              const int* __restrict__ past_seq_lens, const int max_num_blocks_per_seq,
              const int q_stride, const int kv_block_stride, const int kv_head_stride
            ){{
              const int seq_idx = blockIdx.y;
              const int partition_idx = blockIdx.z;
              const int max_num_partitions = gridDim.z;
              const int past_seq_len = past_seq_lens[seq_idx];
            
              const int num_seq_blocks = DIVIDE_ROUND_UP(past_seq_len, {kv_block_size});
            
              const int start_block_idx = 0;
              const int end_block_idx = num_seq_blocks;
              const int num_blocks = end_block_idx - start_block_idx;
            
              const int start_token_idx = start_block_idx * {kv_block_size};
              const int end_token_idx = MIN(start_token_idx + num_blocks * {kv_block_size}, past_seq_len);
              const int num_tokens = end_token_idx - start_token_idx;  
            
              constexpr int THREAD_GROUP_SIZE = MAX({warp_size} / {kv_block_size}, 1);
              constexpr int NUM_THREAD_GROUPS = {num_threads} / THREAD_GROUP_SIZE;
              assert({num_threads} % THREAD_GROUP_SIZE == 0);
              constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP({kv_block_size}, {warp_size});
              constexpr int NUM_WARPS = {num_threads} / {warp_size};
              const int thread_idx = threadIdx.x;
              const int warp_idx = thread_idx / {warp_size};
              const int lane = thread_idx % {warp_size};
            
              const int head_idx = blockIdx.x;
              const int num_heads = gridDim.x;
              const int num_queries_per_kv = num_heads / num_kv_heads;
              const int kv_head_idx = head_idx / num_queries_per_kv;
            
              constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof({scalar_t})), 1);
              using K_vec = typename Vec<{scalar_t}, VEC_SIZE>::Type;
              using Q_vec = typename Vec<{scalar_t}, VEC_SIZE>::Type;
            
              constexpr int NUM_ELEMS_PER_THREAD = {head_size} / THREAD_GROUP_SIZE;
              constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;
            
              const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
              const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;
            
              const {scalar_t}* q_ptr = q + seq_idx * q_stride + head_idx * {head_size};
            
              __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
            
            #pragma unroll
              for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {{
                const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
                q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
              }}
              __syncthreads();
            
              extern __shared__ char shared_mem[];
              float* logits = reinterpret_cast<float*>(shared_mem);
              __shared__ float red_smem[2 * NUM_WARPS];
            
              constexpr int x = 16 / sizeof({cache_t});
              float qk_max = -__FLT_MAX__;
            
              const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
              for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {{
                const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
            
                for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {{
                  const int physical_block_offset = (thread_group_idx + i * {warp_size}) % {kv_block_size};
                  const int token_idx = block_idx * {kv_block_size} + physical_block_offset;
                  K_vec k_vecs[NUM_VECS_PER_THREAD];
            
            #pragma unroll
                  for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {{
                    const {scalar_t}* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride +
                                              physical_block_offset * x;
                    const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
                    const int offset1 = (vec_idx * VEC_SIZE) / x;
                    const int offset2 = (vec_idx * VEC_SIZE) % x;
                    k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * {kv_block_size} * x + offset2);
                  }}
            
                  // QK
                  float qk = scale * Qk_dot<{scalar_t}, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
            
                  // Softmax
                  if (thread_group_offset == 0) {{
                    const bool mask = token_idx >= past_seq_len;
                    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
                    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
                  }}
                }}
              }}
            
            #pragma unroll
              for (int mask = {warp_size} / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {{
                qk_max = fmax(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
              }}
              if (lane == 0) {{
                red_smem[warp_idx] = qk_max;
              }}
              __syncthreads();
            
              qk_max = lane < NUM_WARPS ? red_smem[lane] : -__FLT_MAX__;
            #pragma unroll
              for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {{
                qk_max = fmax(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
              }}
              qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);
            
              float exp_sum = 0.f;
              for (int i = thread_idx; i < num_tokens; i += {num_threads}) {{
                float val = __expf(logits[i] - qk_max);
                logits[i] = val;
                exp_sum += val;
              }}
              exp_sum = block_sum<NUM_WARPS, {warp_size}>(&red_smem[NUM_WARPS], exp_sum);
            
              const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
              for (int i = thread_idx; i < num_tokens; i += {num_threads}) {{
                logits[i] *= inv_sum;
              }}
              __syncthreads();
            
              // Value
              constexpr int V_VEC_SIZE = MIN(16 / sizeof({scalar_t}), {kv_block_size});
              using V_vec = typename Vec<{scalar_t}, V_VEC_SIZE>::Type;
              using L_vec = typename Vec<{scalar_t}, V_VEC_SIZE>::Type;
              using Float_L_vec = typename FloatVec<L_vec>::Type;
            
              constexpr int NUM_V_VECS_PER_ROW = {kv_block_size} / V_VEC_SIZE;
              constexpr int NUM_ROWS_PER_ITER = {warp_size} / NUM_V_VECS_PER_ROW;
              constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP({head_size}, NUM_ROWS_PER_ITER);
            
              float accs[NUM_ROWS_PER_THREAD];
            #pragma unroll
              for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {{
                accs[i] = 0.f;
              }}
            
              {scalar_t} zero_value;
              zero(zero_value);
              for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {{
                const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
                const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
                const int token_idx = block_idx * {kv_block_size} + physical_block_offset;
                L_vec logits_vec;
                from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));
            
                const {cache_t}* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
            
            #pragma unroll
                for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {{
                  const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
                  if (row_idx < {head_size}) {{
                    const int offset = row_idx * {kv_block_size} + physical_block_offset;
                    V_vec v_vec;
                    v_vec = *reinterpret_cast<const V_vec*>(v_ptr, offset);
                    if (block_idx = num_seq_blocks - 1) {{
                      {scalar_t}* v_vec_ptr = reinterpret_cast<{scalar_t}*>(&v_vec);
            #pragma unroll
                      for (int j = 0; j < V_VEC_SIZE; j++) {{
                        v_vec_ptr[j] = token_idx + j < past_seq_len ? v_vec_ptr[j] : zero_value;
                      }}
                    }}
                    accs[i] += dot(logits_vec, v_vec);
                  }}
                }}
              }}
            
              // LV
            #pragma unroll
              for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {{
                float acc = accs[i];
            #pragma unroll
                for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {{
                  acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
                }}
                accs[i] = acc;
              }}
            
              __syncthreads();
            
              float* out_smem = reinterpret_cast<float*>(shared_mem);
            #pragma unroll
              for (int i = NUM_WARPS; i > 1; i /= 2) {{
                int mid = i / 2;
                if (warp_idx >= mid && warp_idx < i) {{
                  float* dst = &out_smem[(warp_idx - mid) * {head_size}];
            #pragma unroll
                  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {{
                    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
                    if (row_idx < {head_size} && lane % NUM_V_VECS_PER_ROW == 0) {{
                      dst[row_idx] = accs[i];
                    }}
                  }}
                }}
                __syncthreads();
            
                if (warp_idx < mid) {{
                  const float* src = &out_smem[warp_idx * {head_size}];
            #pragma unroll
                  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {{
                    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
                    if (row_idx < {head_size} && lane % NUM_V_VECS_PER_ROW == 0) {{
                      accs[i] += src[row_idx];
                    }}
                  }}
                }}
                __syncthreads();
              }}
            
              // Output
              if (warp_idx == 0) {{
                {scalar_t}* out_ptr = out + seq_idx * num_heads * max_num_partitions * {head_size} +
                                      head_idx * max_num_partitions * {head_size} + partition_idx * {head_size};
            #pragma unroll
                for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {{
                  const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
                  if (row_idx < {head_size} && lane % NUM_V_VECS_PER_ROW == 0) {{
                    from_float(*(out_ptr + row_idx), accs[i]);
                  }}
                }}
              }}
            }}
"#
        );

        let (ptx, log) = Ptx::compile(code);
        if !log.is_empty() {
            println!("{log}");
        }
        Self {
          ptx: ptx.unwrap(),
          f: CString::new(name).unwrap(),
          warp_size: warp_size,
          num_threads: num_threads,
        }
    }

    pub fn launch<OutT, QT, KT, VT, BlockTablesT, SeqLensT>(
        &self,
        module: &ModuleSpore,
        out: &mut Tensor<OutT>,
        query: &Tensor<QT>,
        key_cache: &Tensor<KT>,
        value_cache: &Tensor<VT>,
        num_kv_heads: u32,
        scale: f32,
        block_tables: Tensor<BlockTablesT>,
        seq_lens: Tensor<SeqLensT>,
        max_seq_len: u32,
        kv_block_size: u32,
        stream: &Stream,
    ) where
    OutT: DerefMut<Target = [DevByte]>,
    QT: Deref<Target = [DevByte]>,
    KT: Deref<Target = [DevByte]>,
    VT: Deref<Target = [DevByte]>,
    BlockTablesT: Deref<Target = [DevByte]>,
    SeqLensT: Deref<Target = [i32]>,
    {
        let num_seqs = query.shape()[0];
        let num_heads = query.shape()[1];
        let head_size = query.shape()[2];
        let max_num_blocks_per_seq = block_tables.shape()[1];
        let q_stride = query.strides()[0];
        let kv_block_stride = key_cache.strides()[0];
        let kv_head_stride = key_cache.strides()[1];

        let thread_group_size = (self.warp_size / kv_block_size).max(1);
        assert_eq!(head_size % thread_group_size, 0);

        let out_ptr = (out.physical().as_ptr() as isize + out.bytes_offset()) as CUdeviceptr;
        let query_ptr = (query.physical().as_ptr() as isize + query.bytes_offset()) as CUdeviceptr;
        let key_cache_ptr = (key_cache.physical().as_ptr() as isize + key_cache.bytes_offset()) as CUdeviceptr;
        let value_cache_ptr = (value_cache.physical().as_ptr() as isize + value_cache.bytes_offset()) as CUdeviceptr;
        let block_tables_ptr = (block_tables.physical().as_ptr() as isize + block_tables.bytes_offset()) as CUdeviceptr;
        let seq_lens_ptr = (seq_lens.physical().as_ptr() as isize + seq_lens.bytes_offset()) as CUdeviceptr;

        let num_warps: u32 = self.num_threads / self.warp_size;
        let padded_max_seq_len = (max_seq_len + kv_block_size - 1) / kv_block_size * kv_block_size;
        let logits_size = padded_max_seq_len as usize* std::mem::size_of::<f32>();
        let outputs_size = num_warps as usize / 2 * head_size as usize * std::mem::size_of::<f32>();
        let shared_mem_size = logits_size.max(outputs_size);

        let grid = (num_heads, num_seqs, 1);
        let block = (self.num_threads, 1, 1);

        let params: [*const c_void; 12] = [
            (&out_ptr) as *const _ as _,
            (&query_ptr) as *const _ as _,
            (&key_cache_ptr) as *const _ as _,
            (&value_cache_ptr) as *const _ as _,
            (&num_kv_heads) as *const _ as _,
            (&scale) as *const _ as _,
            (&block_tables_ptr) as *const _ as _,
            (&seq_lens_ptr) as *const _ as _,
            (&max_num_blocks_per_seq) as *const _ as _,
            (&q_stride) as *const _ as _,
            (&kv_block_stride) as *const _ as _,
            (&kv_head_stride) as *const _ as _,
        ];

        // // 如果想声明超过48KB的共享内存，必须使用 dynamic shared memory 且要设置 cudaFuncAttributeMaxDynamicSharedMemorySize
        // cuFuncSetAttribute(paged_attention_v1_kernel,
        //   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_mem_size);

        let module = unsafe { module.sprout(stream.ctx()) };
        let kernel = module.get_kernel(&self.f);
        kernel.launch(grid, block,  params.as_ptr(), shared_mem_size as usize, Some(stream))

        // match head_size {
        //   64 => LAUNCH_PAGED_ATTENTION_V1(64),
        //   80 => LAUNCH_PAGED_ATTENTION_V1(80),
        //   96 => LAUNCH_PAGED_ATTENTION_V1(96),
        //   112 => LAUNCH_PAGED_ATTENTION_V1(112),
        //   128 => LAUNCH_PAGED_ATTENTION_V1(128),
        //   256 => LAUNCH_PAGED_ATTENTION_V1(256),
        //   _ => {
        //       panic!("Unsupported head size: {}", head_size);
        //   }
        // }
    }

}

#[test]
fn test_kernel() {
    use cuda::CudaDataType;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context()
        .apply(|_| PagedAttention::new(64, 16, 128, 32, CudaDataType::u16, CudaDataType::u16));
}

