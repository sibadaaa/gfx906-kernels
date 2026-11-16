# 迁移审计（vLLM `csrc/` → gfx906-kernels）

本文记录每个内核家族怎么搬过来的、搬完后和源文件的差异、以及为什么差异只允许出现在
include / 命名空间 / host 侧拼写这三类位置。

## 0. 判定「逐位一致」的依据

数值路径完全由三样东西决定：**设备代码**、**launch 配置**、**编译选项**。

1. 设备代码：`diff` 源文件与目标文件，逐 hunk 审计；所有 kernel 体、`__launch_bounds__`、
   tile/常量、累加顺序、`hipLaunchKernelGGL` 的 grid/block/smem 都未改动。
2. launch 配置：host 侧入口函数体也未改动（除了类型别名解析），所以分支条件与参数一致。
3. 编译选项：同一 `hipcc`、同一 `--offload-arch=gfx906`、`-O3`、`-DUSE_ROCM`，
   未加 `-ffast-math` 或任何改变浮点语义的 flag。

可复现的机器检查（`tests/audit_device_code.py`）：把源文件与迁移文件里的每个
`__global__`/`__device__` 函数体按花括号抓出来、归一化空白后做**多重集比较**，
再把每个 `hipLaunchKernelGGL(...)` 调用点（grid/block/smem/参数）同样比较。当前结果：

```text
device functions checked: 107; all bodies identical
hipLaunchKernelGGL call sites: 63/63 identical
```

最终数值确认靠 `tests/compare_with_vllm.py`：同一输入分别喂 `torch.ops._rocm_C`
与 `torch.ops.gfx906_kernels`，比较原始位模式。2026-11-15 在 4×Z100L、DTK 26.04、
torch 2.11 上跑通 **266/266 用例逐位一致**，覆盖 GEMV（fp16/bf16，含 bias）、
fp16 streaming GEMV、W8A8 int8（gemv/gemm/gemm_pc）、FP8（gemv/gemm/block/W8A16/
smul/quant/quant_x/stage_int8_pc/mm_int8_pc）、RMSNorm 三种、decode attention 内层
（fp16/fp32 分段缓冲 × delayed_merge × descale，另含 int8-KV 变体）、
causal_conv1d decode、GDN sigmoid-gated delta rule（decode 与 packed decode，
输出与原地 state）。原始日志：
`docs/results/ab-gpu-torch211-dtk2604.log`。

参考实现不是用仓库里预编译的 `vllm/_rocm_C.abi3.so`（它和当前源码/环境不匹配，
在 torch 2.11 下连符号都解析不了），而是用**当前 csrc 源码 + 生产编译参数**重编：
`tests/build_reference.sh` 从 vLLM 构建记录（`STMTS`，形如 `rocm_c_rebuild/stmt.txt`）取 DEFINES/FLAGS/INCLUDES，
把 hipify 的头文件映射用 shim 复现，产出 `build/ref/_rocm_C.abi3.so`（线性+注意力）
与 `build/ref/_C_ref.abi3.so`（RMSNorm + GDN）。对比前提：参考实现与适配层必须
用同一个 torch 构建（验证时为 torch 2.11）。

2026-11-15 最终一轮：**266/266 用例逐位一致**，覆盖 9 个家族、22 个算子入口；
新增 int8-KV decode 内层（12）与 GDN packed decode（32，含原地 state）。

**已解决：GDN packed decode 的 1-ulp 偏差。** 早期观察到 `state` 为 fp16 时，
原地 state 有 7–10/262144 个元素差 1 ulp（`out` 始终一致）。排查中先后排除了编译宏全集、
`-mno-fma`、`-ffp-contract=off`、hipcc 的 `-mllvm` 默认参数、`-fno-gpu-rdc`、clang++ 直连，
并且确认被比较的 device code 字节一致——但**当时比错了 kernel**。用 `LD_PRELOAD` 拦截
`hipLaunchKernel` 抓 kernel 名/grid/block/kernarg 后发现真正执行的是
`fused_recurrent_gated_delta_rule_gfx906_m1_native_kernel`，它两边机器码不同
（6092 vs 4648 字节）。根因有两处，都在构建语义：

1. 迁移时为兼容 DTK 6.2 加过本地 `#define __shfl_xor_sync(...)`，**覆盖了 HIP builtin**；
   生产构建没有 shim，而是 `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1` 用 builtin
   （wave32/wave64 语义不同）。
2. 生产构建带 `-DNDEBUG`；不带时 builtin 的 mask 断言在 wave64 + 32-lane 子 warp
   （`__shfl_xor_sync(0xffffffff, x, off, 32)`）上直接触发。

去掉 shim、补上 `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DNDEBUG` 后全部逐位一致。
**编译选项属于数值契约**，这条已写到 README 的构建说明里。

仍未作为数值门禁的只有 `paged_attention`：它在 gfx906 上被编成 `UNREACHABLE_CODE`
（只选 MFMA 变体），vLLM 自己的实现也会触发 device assert，不是可运行路径。

> 为什么不用「对比现网 `.so` 的机器码」：vLLM 检出里预编译的 `_rocm_C.abi3.so`
> 往往是另一个源码快照/编译配置的产物，ISA 对比会把「构建差异」误判成「抽取差异」。
> 源级函数体 + launch 点比较与构建配置无关，更适合作为抽取正确性的门禁；而数值
> 对拍则用「同一份 csrc + 生产编译参数」重编出的参考实现。

## 1. 家族清单

| 目标文件 | 源文件 | 导出（`namespace gfx906`） | 编译 |
|---|---|---|---|
| `src/kernels/bf16_emu_gemv.hip` | `csrc/rocm/bf16_emu_gemv.hip` | `bf16_emu_gemv`, `_meta` | ✅ |
| `src/kernels/w8a8_sdot4.hip` | `csrc/rocm/w8a8_sdot4.hip` | `w8a8_sdot4_gemv/gemm`, 两个 `_meta` | ✅ |
| `src/kernels/sdot4_pc.hip` | `csrc/rocm/sdot4_pc.hip` | `fp8_stage_int8_pc`, `w8a8_sdot4_gemm_pc`, `fp8_emu_mm_int8_pc`, `_meta` ×3 | ✅ |
| `src/kernels/fp8_emu_gemm.hip` | `csrc/rocm/fp8_emu_gemm.hip` | 19 个（gemv/gemm/block/W8A16/quant/dequant/smul + meta） | ✅ |
| `src/kernels/attnd_inner.hip` | `csrc/rocm/attnd_inner.hip` | `gfx906_attnd_decode_inner` | ✅ |
| `src/kernels/attnd_inner_kvint8.hip` | `csrc/rocm/attnd_inner_kvint8.hip` | `gfx906_attnd_decode_inner_kvint8` | ✅ |
| `src/kernels/attention.hip` | `csrc/rocm/attention.hip` | `paged_attention` | ✅ |
| `src/kernels/gfx906_decode_kernels.hip` | `csrc/mamba/gfx906_decode_kernels.hip` | 10 个（conv1d / GDN / packed decode） | ✅ |
| `src/kernels/gfx906_layernorm_kernels.hip` | `csrc/rocm/gfx906_layernorm_kernels.hip` | `rms_norm_gated_gfx906`, `gemma_rms_norm_gfx906`, `gemma_fused_add_rms_norm_gfx906` | ✅ |

未抽离（不属于本模块的「gfx906 计算核」范围）：

* `csrc/activation_kernels.hip` 里的 `dtk_gfx906_fp16_no_packed` 是**上游激活函数**的
  精度 workaround（DTK 26.04 在 gfx906 上 fp16 packed 分支会错），不是独立内核，
  留在 vLLM 的构建里；
* `csrc/quantization/gguf/*` 的 `gfx906_q5k_*` 是 GGUF 量化 matvec 的优化路径，
  生产 FP8 模型不使用；
* `csrc/custom_all_reduce*` 是通信（push-mode one-shot AR），不是计算核；
* `csrc/dtk_rocprim_compat.h` 是构建期兼容头。

纯 gfx906 的 kernel 文件：

* `csrc/rocm/q_gemm_rdna3*`、`moe_q_gemm_rdna3` —— 只在 `gfx1100` 构建；
* `csrc/rocm/skinny_gemms*` —— MFMA ISA，vLLM 在 gfx906 上就把它从构建里剔除；
* `csrc/rocm/quickreduce` —— 通信，不是计算核。

## 2. 逐家族差异

### 2.1 bf16_emu_gemv / w8a8_sdot4 / sdot4_pc / layernorm

差异只有：

```diff
-#include <ATen/dtk_macros.h>
-#include <torch/all.h>
-#include <ATen/hip/HIPContext.h>
-#include <c10/hip/HIPGuard.h>
-#include <c10/util/Float8_e4m3fn.h>
+#include <hip/hip_runtime.h>
+#include "gfx906/detail/aten.hpp"
+#include "gfx906/ops.hpp"
```

加上 `namespace gfx906 { ... }` 包裹 host 入口。layernorm 另把
`core/utils.hpp`/`legacy_dispatch_utils.h`/`vectorization_utils.cuh`/`torch_utils.h`
替换为 `<hipcub/hipcub.hpp>` + compat 里的 dispatch 宏（这些头在 layernorm 里只提供
`CubAddOp` 与 `VLLM_DISPATCH_FLOATING_TYPES`，语义等价）。

### 2.2 fp8_emu_gemm（14 个 device kernel + 3 个 host-only helper）

设备代码零改动。除 include/命名空间外，只有两处**文件级** shim：

1. `__shfl_xor_sync` 4 参数形式 → `__shfl_xor(var, lane, width)`。DTK 6.2 的 HIP 头只在
   `HIP_ENABLE_WARP_SYNC_BUILTINS` 下声明它，而且那版会插 mask 断言；vLLM 原来是从
   `ATen/dtk_macros.h` 拿到 CUDA 写法的。映射与 vendor 的
   `amd_warp_sync_functions.h` 一致，且**保留 `width`**：内核依赖 32 lane 的子 warp 语义。
2. 第二层匿名命名空间（`fused_decode_cfg`、`launch_quant_a`、`dispatch_block_quant`、
   `run_block_quant`）现在嵌套在 `gfx906` 内，仍是文件内部符号；`launch_fp16_gemv_stream_m1`
   被放进新的匿名命名空间以免变成导出符号。函数体未动。

### 2.3 attnd_inner / attnd_inner_kvint8

零设备改动。唯一注意点：`ops_attnd.hpp` 里的声明按真实定义写成 `Tensor&`（源码里全是指针
引用），这样 `.hip` 里的定义与声明由编译器强制对齐。

### 2.4 attention.hip

源文件是平铺的，没有匿名命名空间，所以目标文件里新加了一个匿名命名空间把设备代码包起来
（改变链接性，不改变代码）。`is_navi_gpu()` 随之变成文件内部符号。

额外 shim：vendored `core/cuda_compat.h` 用 CUDA 拼写（`cudaDeviceProp` 等）访问 warp size，
而 HIP 头只有 `hip*` 名字。处理方式是在 include 前后局部 `#define` 映射到 `hip*`，不修改
vendored 头。

### 2.5 gfx906_decode_kernels.hip（GDN / conv1d）

设备代码 md5 与源文件一致（匿名命名空间 1893 行完全相同）。host 侧有 9 处
`getCurrentCUDAStream().stream()` → `getCurrentCUDAStream()`（compat 直接返回
`hipStream_t`），以及 3 处小工具：

* `squeeze0()`：`[1,T,H] -> [T,H]` 的 view（dim0 == 1，等价于 `squeeze(0)`）；
* `int32_tensor()`：用 `hipMemcpy` 建两个缓存的 int32 常量张量；
* `torch::empty_like(x)` 补上 `x.options()`（compat 要求显式 options）。

`__shfl_xor_sync` 3 参数形式 → `__shfl_xor`，原因同 2.2。

## 3. compat 层（`include/gfx906/detail/aten.hpp`）

不是「模拟 torch」，而是**只实现被用到的 15 个 API**：

* `Tensor`：`data_ptr()` / `data_ptr<T>()`、`size/stride/sizes/dim/numel`、
  `scalar_type/dtype/device/options`、`is_cuda/is_contiguous/contiguous`、
  `new_empty/narrow/t`、`defined`；`empty/zeros/empty_like` 用可替换 allocator 分配。
* `at::cuda::getCurrentCUDAStream()` 直接返回 `hipStream_t`（所以源码里 `.stream()`
  的那几处要改拼写）；`CUDAGuard`/`OptionalCUDAGuard` 会 `hipSetDevice` 并在析构时恢复。
* `at::Half` = `__half`，`at::BFloat16` = `__hip_bfloat16`；
  `c10::Float8_e4m3fn` 是从 PyTorch 头文件**逐行复制**的转换（`detail/fp8_e4m3.h`），
  因为 block quantizer 在 device 侧用它，位级必须一致。
* `TORCH_CHECK` → 抛 `gx::Error`，由 C ABI 捕获并转成 `gx_status` + `gx906_last_error()`。
* `VLLM_DISPATCH_FLOATING_TYPES` / `VLLM_DISPATCH_HALF_TYPES`。

已知代价：host 代码仍写着 `torch::Tensor`（通过别名解析到 `gx::Tensor`）。这是有意为之——
让 `diff` 源文件时只剩 include/命名空间两类噪声，便于以后从 vLLM 主线继续搬改动。

## 4. 内存与所有权

* 输入永远是 view：不拷贝、不转置、不反量化。
* 输出由库分配：默认 `hipMalloc`；`gx906_set_allocator` 可换成调用方的池。
  torch 适配层安装 `raw_alloc/raw_delete`，并用 `at::from_blob` + 释放回调把
  `gx_tensor` 包成 `at::Tensor`，因此输出也走 torch 的 caching allocator，不会出现
  「库内存」和「torch 内存」两套账。
* `contiguous()` 的快路径按行 `hipMemcpy2D`（最后一维单位步长）；否则走 `copy_strided.hip` 的通用 strided 拷贝内核（≤ 8 维）。超过 8 维显式报错，不会静默算错。

## 4.1 torch 适配层的两处修正（加载时发现的真实缺陷）

1. 多返回值：C++ 返回 `std::vector<at::Tensor>` 会被 torch 推断成 `Tensor[]`，与 schema
   里写的 `(Tensor, Tensor, Tensor)` 不符，`load_library` 直接抛
   `Inferred operator schema ... doesn't match`。改成 `std::tuple` 后一致。
2. `out_dtype` 必须是 schema 里的 `ScalarType`（C++ 侧 `at::ScalarType`），不能写 `int`，
   否则 vLLM 侧传 `a.dtype` 的调用点无法直接替换。

这两处都是在本机 `torch.ops.load_library()` 真实注册时暴露并修掉的（不需要 GPU）。

## 5. 还未做的（刻意划边界）

* **Python 派发层**：`vllm/model_executor/kernels/linear/scaled_mm/gfx906_*.py` 里的
  M 阈值、cfg 选择、fp16 prefill 物化、`_device_lut` 等，是引擎策略，不属于 kernel。
  接别的引擎时应按该引擎的 layout 重写。
* **Triton 内核对 gfx906 的分支**（`vllm/v1/attention/ops/triton_unified_attention.py`
  等）没有抽离；它们依赖自编 Triton，不是 HIP 模块的一部分。
* **数值门禁的运行环境**：`tests/compare_with_vllm.py` 需要 4×Z100L、DTK 26.04、
  torch 2.11（与参考扩展同版本）。重建参考实现的脚本是 `tests/build_reference.sh`。
