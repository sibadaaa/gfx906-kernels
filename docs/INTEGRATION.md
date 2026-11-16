# 对接指南（非 vLLM 引擎）

目标读者：要把这套 kernel 嵌进自己的推理引擎（C/C++、Rust、Python 或 torch 引擎）的人。
只讲**契约**：怎么传内存、权重什么布局、哪些量和数值由谁负责。

## 0. 三条接入路径

| 路径 | 适用 | 入口 |
|---|---|---|
| C ABI | 任意语言（C/Rust/Go/ctypes） | `include/gfx906/gfx906.h` + `libgfx906_kernels.so` |
| C++ API | C++ 引擎，想要零成本、无描述符转换 | `include/gfx906/ops.hpp`（`gx::Tensor` 视图） |
| torch 适配层 | vLLM / SGLang 这类 torch 引擎 | `libgfx906_torch.so` → `torch.ops.gfx906_kernels.*` |

三条路径共用同一份内核与同一个 allocator/stream 机制；C ABI 是稳定面，C++ API 与它一一对应。

## 1. 运行时契约

```c
gx906_set_stream(hip_stream);   /* 所有 kernel 用它，不做隐式同步 */
gx906_set_device(0);
```

* **stream**：库不创建、不销毁、不同步 stream。调用返回只代表 kernel 已入队。
* **device**：kernel 按张量描述符里的 `device` 校验；分配用 `gx906_set_device` 的当前设备。
* **allocator**：默认 `hipMalloc`；引擎可换成自己的池：
  ```c
  gx906_set_allocator(alloc_cb, free_cb, pool_ctx);
  ```
  输出内存归库引用，`gx906_tensor_release()` 归还；输入视图永远不释放。
* **错误**：返回 `gx_status`；`gx906_last_error()` 给线程局部文本。不会 abort，也不会静默返回半成品：
  形状/步长/dtype/空指针都会在 launch 之前被 `TORCH_CHECK` 等价检查拦下。

## 2. 通用张量约定

* `gx_tensor.strides` **以元素为单位**（与 torch 一致），`NULL` 表示紧凑排布。
* dtype 编码与 torch `ScalarType` 相同（`GX_DTYPE_*`），便于从 torch 直接传。
* 权重永远按 **checkpoint 布局 `[N, K]`** 传入：不转置、不反量化、不重排。
* `M` 是 token 数（decode 时 1..8）。**选择 gemv/gemm 是引擎的责任**（vLLM 的阈值在
  `vllm/model_executor/kernels/linear/scaled_mm/gfx906_*.py`），本模块不内建 M 派发策略。

## 3. 线性层

### 3.1 FP8（e4m3）权重，per-tensor/per-row scale

```
a   [M, K]  fp8 e4m3
w   [N, K]  fp8 e4m3
a_s [M] 或 [1]   fp32     (per-token 或 per-tensor)
w_s [N] 或 [1]   fp32     (per-channel 或 per-tensor)
lut [256] fp16           fp8 code -> fp16 的展宽表
out [M, N] fp16/bf16
```

`lut` 用 `gx906_fp8_lut_fill()` 生成（一次性，256 项），或用同样的展宽规则自己构造。
数学契约：`out = ((sum_k a*w) * a_s) * w_s + bias`，fp32 累加，最后 cast 到 `out_dtype`。

### 3.2 FP8 block-scaled（DeepSeek 风格）

```
a_s [M, ceil(K/block_k)]                 fp32
w_s [ceil(N/block_n), ceil(K/block_k)]   fp32
```

`block_k` 必须整除 `K`（vLLM 生产为 128×128）。W8A16 变体不量化激活：
`a` 直接是 fp16/bf16，`w` 仍是 fp8 + block scale，因此没有 `a_s`。

### 3.3 W8A8 INT8（`v_dot4_i32_i8`）

```
xq [M, K] int8, w [N, K] int8, x_s [M]或[1], w_s [N]或[1], out [M,N]
```

适合 prefill 大 M（gfx906 上比 rocBLAS int8 高一个量级）；decode 小 M 用 3.1/3.2。

### 3.4 稠密 fp16/bf16 GEMV（M ≤ 8）

```
x [M, K], w [N, K], out [M, N]，同 dtype；bias 可选 [N]
route: 0=auto（fp16→dot2, bf16→fp32）, 1=fp32, 2=v_dot2_f32_f16
```

`range_safe` 只对 bf16→fp16 rebase 路由有意义：调用方要先确认权重动态范围，否则默认 fp32 路由。

## 4. 归一化

```
input [T, H]（或更高维，最后一维 H），gate [T, H]，weight [H]
out   [T, H]
```

`norm_before_gate` 决定 gate 的 SiLU 乘在归一化前还是后（Gemma 系 vs 普通 gated RMSNorm）。
非连续输入会先物化（行拷贝快路径，或通用 strided 拷贝内核）。

## 5. 注意力

### 5.1 paged attention（legacy）

布局与 vLLM 一致：`query [num_seqs, num_heads, head_size]`，
`key_cache [num_blocks, num_heads, head_size/x, block_size, x]`，
`value_cache [num_blocks, num_heads, head_size, block_size]`，
`block_tables [num_seqs, max_blocks]`，`seq_lens [num_seqs]`。
`kv_cache_dtype` 支持 `"auto"` / `"fp8"`。分页缓冲 `exp_sums/max_logits/tmp_out`
由调用方按 split-KV 约定分配。

### 5.2 gfx906 decode 内层（生产路径）

这是本模块最有价值的一块（S53/B832，140 GB/s 量级），但它**不是通用注意力**：

* 支持子集在 kernel 内强制：`num_seqs==1`、causal、fp16 Q/K/V、`head==256`、GQA6、
  `block_size % 16 == 0`（生产是 832-token hybrid page，K|V 打包、token stride 512）、
  16 字节对齐的 KV 步长；不支持 sinks/alibi/softcap/qq-bias/sliding-window/fp8-KV。
* 分段缓冲（`segm_output/segm_max/segm_expsum`）与 `reduce_segments` 的契约和
  vLLM 的 Triton 3D kernel 完全一致，**合并（reduce）那一步仍由调用方完成**。
* int8 变体（`..._kvint8`）的 KV page 是
  `[blocks, nkv, block_size, 2*(head+pad)]`，每个 head 后内联 fp32 scale，
  Q 在 prologue 内按 (token, head) 量化。

> 数值说明：该内核的累加顺序**有意**不同于 Triton 版本，生产验收门是
> `|d| <= 0.011` vs SDPA fp32，不是逐位等于 Triton。它逐位等于 vLLM 自己的
> HIP 实现（本模块的抽取源）。

## 6. GDN / Mamba

`causal_conv1d_*` 与 `fused_*gated_delta_rule_*` 的形状契约直接沿用 vLLM 调用点
（`vllm/model_executor/layers/mamba/gdn_linear_attn.py`、`qwen3_5.py`），
包括 `state_indices`、`cu_seqlens`、`num_accepted_tokens`（MTP）语义。
所有形状在 kernel 入口校验；建议照抄 vLLM 的调用参数而不是自行推导。

## 7. 接入清单

1. 建 stream，`gx906_set_stream`；如需自有显存，`gx906_set_allocator`。
2. 权重保持 checkpoint 布局；fp8/int8 权重不要预展宽——展宽是 kernel 内 LDS 查表完成的。
3. 生成一次 256 项 fp8 LUT（`gx906_fp8_lut_fill`）。
4. 按 M 选择 gemv/gemm 变体（策略在引擎侧）。
5. 每次调用检查 `gx_status`，把 `gx906_last_error()` 打进引擎日志。
6. 在宿主上用 `tests/compare_with_vllm.py` 对拍，再跑你自己的端到端基准。
