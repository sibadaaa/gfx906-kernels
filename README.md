# gfx906-kernels

海光 DCU Z100 / Z100L（`gfx906`，Vega20 级，60 CU，无 MFMA、无原生 FP8）的**计算内核独立模块**。

从 [`../csrc/rocm`](../csrc/rocm) 与 [`../csrc/mamba`](../csrc/mamba) 中抽离，目标只有一个：
**让别的推理引擎不依赖 vLLM 也能直接调用这套 kernel**。

```
                 ┌──────────────────────────┐
                 │  你的引擎 (C/C++/Rust)   │
                 └────────────┬─────────────┘
                              │  gfx906.h  稳定 C ABI
                 ┌────────────▼─────────────┐
                 │  libgfx906_kernels.so    │  ← hipcc 编译，链 libamdhip64
                 │  host 胶水 + HIP 内核     │
                 └────────────▲─────────────┘
                              │  可选
                 ┌────────────┴─────────────┐
                 │ torch 适配层 (薄)         │  torch.ops.gfx906_kernels.*
                 └──────────────────────────┘
```

## 1. 设计约束（为什么这样抽）

| 约束 | 做法 |
|---|---|
| **数值必须和现网 vLLM 路径逐位一致** | 设备代码、launch 几何、tile 常数、累加顺序**一字不改**地搬过来；只替换 host 侧的 include 与命名空间 |
| **不能依赖 torch** | 自己实现一个 ~600 行的 ATen 子集（`include/gfx906/detail/aten.hpp`），host 代码用 `torch::Tensor` 的写法可以直接编译 |
| **别的引擎要能接** | 对外只暴露 C ABI（`include/gfx906/gfx906.h`）：`gx_tensor` = 裸指针 + sizes/strides + dtype + device，显式传 HIP stream |
| **不拷贝权重** | 全部 kernel 按 checkpoint 布局原地读（fp8 权重 `[N, K]`、int8 权重 `[N, K]`、KV page 布局不变） |
| **输出内存归调用方** | 默认 `hipMalloc`，也可通过 `gx906_set_allocator` 换成引擎自己的池；torch 侧用 caching allocator |

逐位一致的验证方式：同一份 device 代码 + 同一个 `hipcc` + 同一个 `--offload-arch=gfx906`，
再对拍 `_rocm_C` 与 `gfx906_kernels` 两个算子实现的输出（`tests/compare_with_vllm.py`）。

## 2. 目录

```
gfx906-kernels/
  include/gfx906/
    gfx906.h            # 稳定 C ABI（唯一对外的头）
    ops.hpp             # 可选 C++ API（gx::Tensor，零成本）
    ops_linear.hpp      # 线性层：GEMV / W8A8 int8 / FP8-emu / W8A16
    ops_attnd.hpp       # decode attention inner loop（fp16 / int8 KV）
    ops_attention.hpp   # paged attention（legacy 后端）
    ops_mamba.hpp       # GDN / causal_conv1d
    ops_norm.hpp        # fused RMSNorm 家族
    detail/aten.hpp     # torch-free ATen 子集（内部实现细节）
    detail/fp8_e4m3.h   # fp8 e4m3fn 转换，与 PyTorch 位级一致
  src/kernels/*.hip     # 抽离后的内核（设备代码与 vLLM 源一致）
  src/capi/             # C ABI 实现（描述符 ↔ gx::Tensor）
  src/runtime.cpp       # allocator / stream / device 钩子
  src/vendor/           # 从 vLLM 取的最小编译依赖（Apache-2.0）
  torch/gfx906_torch.cpp# 可选 torch 适配层
  python/gfx906_kernels # ctypes + torch ops 的 Python 封装
  tests/                # 编译门禁 + 逐位对拍
  tools/build.sh        # 构建 libgfx906_kernels.so
```

## 3. 构建

要求：DTK 26.04（HIP 6.2/6.3 均可）、`hipcc`、gfx906 目标。

```bash
cd gfx906-kernels
export DTK_ROOT=/opt/dtk          # DTK 26.04 安装树；非默认路径时改这里
tools/build.sh
# 产物: build/libgfx906_kernels.so

# 可选：torch 适配层（只在 torch 引擎里需要）
tools/build_torch.sh
# 产物: build/libgfx906_torch.so
```

只编 `gfx906`，不链接 torch，也不需要 Python。库的依赖仅 `libamdhip64` 与 C++ runtime。

> **编译选项是数值契约的一部分。** 这些 flag 必须与 vLLM 生产构建一致，否则个别 kernel
> 会出现 1 ulp 差异：`-mno-fma -fno-gpu-rdc -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DNDEBUG`。
> 特别是 **不要** 自己定义 `__shfl_xor_sync` 宏——要用 HIP builtin，因为生产构建用的是它
> （builtin 的 wave32/wave64 语义与旧 3 参数 intrinsic 不同）。
torch 适配层是独立的 `.so`，由 `torch.ops.load_library()` 加载，用哪个 Python/torch 构建
就链接哪个（`PYTHON=` 可指定解释器）。

## 4. 用 C ABI 对接（推荐给非 torch 引擎）

```c
#include "gfx906/gfx906.h"

// 1. 绑定调用方的 stream / device
gx906_set_stream((void*)my_hip_stream);
gx906_set_device(0);

// 2. 可选：用引擎自己的内存池
gx906_set_allocator(my_alloc, my_free, my_pool);

// 3. 描述张量（非拥有视图，不拷贝）
gx_tensor x;  gx906_tensor_view(&x,  x_ptr,  (int64_t[]){M, K}, NULL, 2, GX_DTYPE_F16, 0);
gx_tensor w;  gx906_tensor_view(&w,  w_ptr,  (int64_t[]){N, K}, NULL, 2, GX_DTYPE_F16, 0);

// 4. 调用；输出由库分配
gx_tensor out;
gx_status st = gx906_bf16_emu_gemv(&x, &w, NULL, /*nthreads=*/256, 0, 0, &out);
if (st != GX_OK) { fprintf(stderr, "%s\n", gx906_last_error()); }

// 5. 用完释放（只对库返回的输出需要）
gx906_tensor_release(&out);
```

要点：

* `gx_tensor.strides` 以**元素**为单位（与 torch 一致），传 `NULL` 表示紧凑排布。
* 权重布局就是 checkpoint 布局（`[N, K]`），**不要转置/反量化**；fp8 kernel 自己查 LUT 展宽。
* fp8 的 256 项 LUT 由 `gx906_fp8_lut_fill` 生成（`fp16[256]`），或调用方自己用同样的展宽表。
* 所有 kernel 读 `gx906_set_stream` 设置的 stream，不会隐式同步。
* 错误：返回 `gx_status`，详情用 `gx906_last_error()`（线程局部）。

Python（ctypes）示例见 `python/gfx906_kernels/_ffi.py`。

## 5. 用 torch 对接（vLLM / SGLang 这类引擎）

```python
from gfx906_kernels import load_torch_ops
load_torch_ops()          # 注册 torch.ops.gfx906_kernels.*

y = torch.ops.gfx906_kernels.bf16_emu_gemv(x, w, None, 256, 0, False)
```

op 的 schema 与 vLLM 的 `_rocm_C` 一一对应（`gfx906_attnd_decode_inner`、
`w8a8_sdot4_gemm`、`fp8_emu_gemm_block_w8a16` …），所以迁移只是换命名空间：

```python
# before: torch.ops._rocm_C.fp8_emu_gemv(...)
# after:  torch.ops.gfx906_kernels.fp8_emu_gemv(...)
```

适配层只走 C ABI，torch 侧只负责：张量描述符、当前 stream、caching allocator、
把返回的 `gx_tensor` 包成 `at::Tensor`（`from_blob` + 释放回调）。

## 6. 迁移状态

| 家族 | C ABI | torch 适配层 | 编译 |
|---|---|---|---|
| dense fp16/bf16 GEMV | ✅ | ✅ | ✅ |
| W8A8 int8 (v_dot4) | ✅ | ✅ | ✅ |
| FP8-emu GEMM/GEMV/W8A16/quant/stage | ✅ | ✅ | ✅ |
| decode attention inner（fp16 / int8 KV） | ✅ | ✅ | ✅ |
| paged attention | ✅ | ✅ | ✅ |
| GDN / causal_conv1d | ✅ | ✅ | ✅ |
| fused RMSNorm | ✅ | ✅ | ✅ |

* C ABI：`gfx906.h` 声明 43 个入口，`libgfx906_kernels.so` 中 43/43 已定义（`nm` 校验）。
* torch 适配层：32 个 op，schema 与 vLLM `_rocm_C` 一致（含 `ScalarType out_dtype`
  与 `Tensor!` 原地参数），已在本机 `torch.ops.load_library()` 注册通过。
* 全部 9 个 kernel 文件通过 `tests/compile_gate.sh`（gfx906 目标、`-O3`、无 GPU）。

`RDNA3`（`gfx1100`）的 `q_gemm_rdna3*` / `moe_q_gemm_rdna3` 与 `skinny_gemms*`
不属于 gfx906 路径（vLLM 在 gfx906 上就不编译它们），未抽离。

## 7. 验证

```bash
# 0) 一条命令跑完所有不需要 GPU 的检查
tests/run_all.sh

# 1) 编译门禁（不需要 GPU）
tests/compile_gate.sh

# 2) 设备代码审计（不需要 GPU）
python3 tests/audit_device_code.py --verbose

# 3) fp8 转换与 PyTorch 逐位对拍（不需要 GPU，只需 torch CPU）
tools/build_tests.sh && python3 tests/test_fp8_conversion.py

# 4) 与 vLLM 源码构建的参考实现对拍（需要 GPU）
# 需要本地 vLLM 检出与其构建记录（VLLM_REPO / STMTS / TORCH_DIR 可覆盖）
tests/build_reference.sh          # 用生产编译参数重编 csrc 里的参考实现
VLLM_DCU_FP8_GEMV_WA=1 \
GX906_REF_LIB=build/ref/_rocm_C.abi3.so \
GX906_REF_C_LIB=build/ref/_C_ref.abi3.so \
python tests/compare_with_vllm.py
```

> 参考实现与 `libgfx906_torch.so` 必须与运行对拍的 torch **同一个**（验证时是
> torch 2.11）。换 torch 后先重新跑 `tools/build_torch.sh` 和
> `tests/build_reference.sh`，否则 `load_library` 会因 ABI 不符失败。

已跑出的结果（4× Z100L，DTK 26.04 / torch 2.11 / gfx906）：

| 检查 | 结果 |
|---|---|
| **GPU 数值对拍（原始位模式）** | **266/266 逐位一致** |
| 　· bf16/fp16 GEMV | 48 |
| 　· fp16 streaming GEMV（lm_head） | 3 |
| 　· W8A8 int8（gemv/gemm/gemm_pc） | 22 |
| 　· FP8（gemv/gemm/block/W8A16/smul/quant/stage/mm_int8_pc） | 31 + 4 |
| 　· RMSNorm / GemmaNorm / fused-add | 42 |
| 　· gfx906 decode attention 内层 | 24 |
| 　· causal_conv1d decode | 32 |
| 　· GDN sigmoid-gated delta rule（decode 16 + packed decode 32） | 48 |
| 　· decode attention int8-KV（S81 布局） | 12 |
| 设备函数体逐字一致（9 个文件） | 107/107 |
| `hipLaunchKernelGGL` 调用点（含 grid/block/smem） | 63/63 |
| fp8 e4m3fn 与 PyTorch 转换 | 4352/4352 位级一致 |
| C ABI 从纯 C99 编译器可用 | ✅（`examples/minimal_gemv.c`） |
| C ABI 声明/定义一致 | ✅ 43/43 |

说明：

* `paged_attention`（legacy 后端）在 gfx906 上编译为 `UNREACHABLE_CODE`（仅 MFMA 变体），
  两套实现都会触发 device assert，因此默认跳过；gfx906 的生产注意力路径是
  `gfx906_attnd_decode_inner*`（上表已覆盖）。
* 早期曾观察到 GDN `packed_decode` 原地 fp16 state 有 1 ulp 差异；根因是**构建语义**
  而非抽取：本地 `__shfl_xor_sync` shim 覆盖了 HIP builtin，且缺少生产构建的
  `-DNDEBUG`（builtin 的 mask 断言在 wave64 + 32-lane 子 warp 上会触发）。
  改为与生产一致的 `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DNDEBUG` 后，266 个用例全部
  逐位一致。详见 MIGRATION §5。

`compare_with_vllm.py` 会对每个家族取真实形状（decode M=1..8、prefill M=512/4096、
GQA decode、int8 KV 等），逐元素比较原始位模式，并给出首个不等的下标与两边值。

不装 GPU 也能做的两项：

```bash
# C ABI 能否加载（返回 ABI 版本与 build id）
.venv/bin/python - <<'PY'
import ctypes
lib = ctypes.CDLL("gfx906-kernels/build/libgfx906_kernels.so")
lib.gx906_abi_version.restype = ctypes.c_int32
lib.gx906_build_id.restype = ctypes.c_char_p
print(lib.gx906_abi_version(), lib.gx906_build_id().decode())
PY

# 反向依赖检查：核心库不应引用 torch/c10
nm -D -u gfx906-kernels/build/libgfx906_kernels.so | grep -c "torch\|c10"   # -> 0
```

## 8. 边界与后续

* **只含 kernel**，不含 vLLM 的 Python 派发/量化预处理（`Gfx906Fp8ScaledMMLinearKernel`、
  `_device_lut`、prefill 物化策略）。这些是引擎侧策略，接别的引擎时应按该引擎的
  layout/量化约定重写，而不是照搬。
* **不保证跨 DTK 版本**。与 vLLM 分支一样，DTK/torch 变更是风险点；库本身只依赖 HIP。
* **数值门禁**依赖本机 `_rocm_C`，所以对拍脚本只在打过 gfx906 补丁的 vLLM 环境里有意义。
* 详细迁移记录与逐项差异审计见 `docs/MIGRATION.md`；接入别家引擎的契约（布局、scale、LUT、
  KV page、GDN 参数）见 `docs/INTEGRATION.md`。

## 9. 许可证与归属

- 本仓库整体按 **Apache License 2.0** 发布，全文见 [LICENSE](LICENSE)。
- HIP 内核抽离自 [vLLM](https://github.com/vllm-project/vllm)（Apache-2.0），设备代码保持原样；
  `include/gfx906/detail/fp8_e4m3.h` 的 fp8 转换来自 PyTorch（BSD-3-Clause）。
  完整归属见 [NOTICE](NOTICE)；各文件保留原始版权头。
- `src/vendor/` 下的头文件为 vLLM 构建所需的最小依赖子集（Apache-2.0，部分源自
  NVIDIA/FasterTransformer）。

## 10. 贡献

欢迎 issue / PR。提交前请至少跑通 `tests/run_all.sh`（无需 GPU）；涉及数值路径的改动
还应跑 `tests/compare_with_vllm.py`（需要 Z100L 机器），并在 PR 里附上对拍结果。
