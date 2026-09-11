# FlagGems #5739：广播 pointwise 的 block pointer 零步长编译崩溃 — 分析与修复报告

- **Issue**: [flagos-ai/FlagGems#5739](https://github.com/flagos-ai/FlagGems/issues/5739)
- **代码基线**: FlagGems v5.3.5（`utils/pointwise_dynamic.py` 与 issue 引用结构一致）
- **实验环境**: Ascend 910（8 卡，共享）、CANN 8.5、torch 2.10.0+cpu、torch_npu 2.10.0、
  triton-ascend 3.5.1（flagtree 0.6.1+ascend3.5）、Python 3.11.15
- **报告日期**: 2026-09-11

---

## 1. 问题

### 1.1 现象与"可能会"的真相

issue 报障环境（triton 3.2 / CANN 8.5 / flagtree 0.6.0）中广播 pointwise 报
`strides must not be zero`（ConvertLinalgRToBinary pass）。**本机默认配置不触发**
（15 用例广播矩阵全过），原因：本仓库 `codegen_config_utils.py` 中
`vendors.ASCEND` 的 `prefer_block_pointer=False`。

触发需要同时满足：**广播输入 + block pointer 路径被激活**。后者在三种情况下
成立：老版 triton-ascend 的默认/启发式不同（issue 环境）、NVIDIA 后端（默认
True）、用户显式开启。本机通过强制 `prefer_block_pointer=True` 完整复现。

### 1.2 根因链（复现 + 隔离实验坐实）

```
广播 pointwise: add((128,1), (1,128))
  → pointwise_dynamic.prepare_args (慢路径):
      broadcasted_stride(shape, stride, task_shape) 把广播维 stride 置 0
      → StridedBuffer(零拷贝虚拟广播)
  → 生成的 kernel 走 block pointer (tl.make_block_ptr)
      → memref descriptor 要求步长非零
      → TritonToLinalg pipeline 拒绝:
          ConvertTritonIRToLinalgIR (triton-ascend 3.5.1, 本机)
          ConvertLinalgRToBinary    (triton 3.2, issue 环境 — 同 pipeline 族)
```

隔离实验（`tests/probe_isolate.py`，bptr 强制开启）：

| 输入 | 结果 |
|---|---|
| (128,128)+(128,128) 非广播 | OK，数值正确 |
| (128,1)+(1,128) 广播 | MLIRCompilationError |

**0 步长是唯一变量。**

### 1.3 层次结论

| 层 | 是否有问题 | 依据 |
|---|---|---|
| flag_gems `pointwise_dynamic` 无广播守卫 | **有（本修复）** | 0 步长合法产生，但 bptr 承接不住，缺分流 |
| flag_gems `prepare_args` 单例污染 | **有（顺带修）** | INT32_MAX 守卫原地改共享 config（PR #5668 也指出） |
| triton-ascend TritonToLinalg | 有（上游，可选修） | memref descriptor 不支持 0 步长（NVIDIA 走 expand 表示所以不崩） |
| `broadcasted_stride` / `StridedBuffer` 本身 | 无 | 线性寻址下 0 步长语义完全正确 |

### 1.4 交叉证据

PR #5668（NVIDIA，未合并）：同根因（广播 0 步长 × block pointer），NVIDIA 侧
报错 `Last dimension must be contiguous`；其修复方案（广播检测 + 关 bptr +
config 拷贝）与本修复同构，本工作独立验证了该方案在 Ascend 成立，并额外修了
单例污染的完整链路。

---

## 2. 修复

### 2.1 改动（`utils/pointwise_dynamic.py`，均带 `wt-2026-09-11-fix` 署名）

1. **广播守卫**：`prepare_args` 慢路径检测 `item.shape != task_shape`（任一
   operand 被广播）→ `need_no_block_pointer=True`
2. **per-call config**：返回前用 `dataclasses.replace(self.config,
   prefer_block_pointer=False)` 生成调用级拷贝；`_call_real_impl` 用它做本次
   `instantiate`（try/finally 恢复），共享单例永不被改
3. **INT32_MAX 守卫同改**：原来直接 `self.config.prefer_block_pointer = False`
   （全局污染），现并入同一 per-call 机制
4. 快路径（`use_fast_path`，同形连续）零改动——不产生 0 步长

### 2.2 为什么在 flag_gems 层修而不是 triton-ascend 层

上游修法是让 TritonToLinalg 接受 0 步长（或用 expand 表示），但涉及 MLIR
语义讨论周期长；flag_gems 分流十几行、正确性等价（线性寻址 0 步长天然合法）、
对 bptr 正常路径零影响。两层不冲突，上游支持后分流自然退化为无害冗余。

---

## 3. 实验结果（全部真实 NPU 执行）

| # | 验证项 | 方式 | 结果 |
|---|---|---|---|
| 1 | 根因复现 | `probe_force_bptr.py` 强制 bptr + issue 原例 | MLIRCompilationError（ConvertTritonIRToLinalgIR） |
| 2 | 根因隔离 | `probe_isolate.py` | 非广播 OK / 广播崩——唯一变量是 0 步长 |
| 3 | TDD 红 | `test_red.py`（修复前，强制 bptr） | **10/12 FAIL**（mul 意外过：其 schema 触发快路径） |
| 4 | TDD 绿 | 同上（修复后） | **12/12 PASS**（5 广播位置 × 5 算子 + bptr 正常路径 + 数值） |
| 5 | 守卫副作用 | `guard_sidecheck.py` | 单例不污染 ✓ / 非广播仍生成 `_bptr_` kernel ✓ / 广播走线性 kernel ✓ |
| 6 | 默认配置矩阵 | `repro_matrix.py` 15 用例 | 15/15（修复是 no-op，行为不变） |
| 7 | 仓库回归 pointwise_dynamic | pytest --ref cpu | **1232 passed / 0 failed** |
| 8 | 仓库回归 add | pytest --ref cpu | 1010 passed |
| 9 | 回归裁定（mul 44f） | git stash 原版对照 | 原版 43f——存量 inf 溢出（mul kernel `Greatest abs diff: inf`），与本修复无关 |
| 10 | 性能护栏 | `perf_guard.py` | 默认配置 no-op；强制 bptr 下广播 1081µs vs 非广播 19404µs（读内存少 4000x，合理） |

---

## 4. 诚实自查

### ⚠️ 已声明的近似

1. **issue 环境未实机**：复现在 triton-ascend 3.5.1 强制 bptr 下取得（等价
   失败，同 pipeline 族不同 pass 名）；triton 3.2 环境推断同修未验证。
2. **性能数据在共享机器测得**（外部 benchmark 进程占 AICore 99%），绝对值有
   干扰，只用相对结论。
3. **TDD 红的 mul 用例意外通过**：广播 mul 走了快路径（其 schema 组合满足
   use_fast_path 条件），说明守卫覆盖面（慢路径）与快路径分界需要上游文档
   化；本次未深挖分界的精确条件。

### 🚫 有意跳过

4. **其他 backend 的 pointwise_dynamic 副本**（cambricon/sunrise/kunlunxin
   等独立文件）未改——同模式可平移，但无实机验证，留上游。
5. **mul 存量 inf 溢出**未修（独立 bug，值得单独 issue）。

### ❌ 过程中修正过的错误

6. `guard_sidecheck.py` 初版用文件 glob 时序判断 kernel 新增，因 `__pycache__`
   复用全部落空；改为直接检查 code_cache 目录的 kernel 变体命名
   （`rank_2__t512` 线性 vs `rank_1_bptr` bptr）得证。
7. 初跑复现矩阵死于 `randn(0)` 的已知零 grid bug（randn.py:145，liftfreshFIX
   报告记录过）——移除空张量用例后矩阵干净。测试脚本的输入构造要避开已知
   崩溃点，否则归因混淆。

---

## 5. perf 数据附录（路线参考，见 README 对话背景）

用户追问"既然原生快，为什么不分发回原生 / Triton 能否写得不差"。实测
（`tests/handwrite_bench.py` / `host_overhead.py`，共享机器，中位数）：

| 场景 | 原生 torch_npu | 手写 Triton flat | 结论 |
|---|---|---|---|
| 128MB 大张量 add | 5210 µs | **3235 µs（0.62x）** | Triton 反超，无天花板惩罚 |
| 32MB 中张量 | 793 µs | 4483 µs | host 开销主导 |
| Triton 纯 host launch 链路 | — | **~35 µs/次** | Python 打包是固定成本 |
| flag_gems codegen 端到端（4MB） | 76-760 µs | 250-1350 µs | 框架 host 层再叠加 |

**结论**：中小张量差距主要在 Python host 链路（原生 C++ 直连 aclnn），
kernel 本体带宽利用率在大形状已证明。开源路线建议：① correctness 分流
（本修复）；② host 链路减负（C++ launch / wrapper 缓存）；③ 形状感知
fallback 阈值（利用现成 `use_gems(exclude=...)` 机制，打赢的区间开 Triton）；
④ 部署态 AOT 打包 npubin（对冷启动重要，对稳态性能帮助有限——triton 缓存
本已按参数 hash 命中）。

## 6. 复现 / 验证命令速查

```bash
cd /root/pointwiseFIX
ASCEND_LAUNCH_BLOCKING=1 python tests/repro_5739.py
ASCEND_LAUNCH_BLOCKING=1 python tests/test_red.py          # 12/12
ASCEND_LAUNCH_BLOCKING=1 python tests/guard_sidecheck.py
ASCEND_LAUNCH_BLOCKING=1 python tests/repro_matrix.py      # 15/15
ASCEND_LAUNCH_BLOCKING=1 python tests/perf_guard.py
cd /root/FlagGems && python -m pytest tests/test_pointwise_dynamic.py tests/test_add.py -q --ref cpu
```

## 7. 建议后续

1. `pointwise_dynamic.patch` 提交上游，引用 issue #5739，并关联 PR #5668
   （同根因 NVIDIA 侧，我们的 Ascend 实证可作其佐证）。
2. 上游把"单例污染"独立成一个 issue（影响所有 vendor 的正确性，与本 bug 无关）。
3. triton-ascend 考虑 TritonToLinalg 对 0 步长的支持或更清晰的报错。
4. mul 的存量 inf 溢出单独排查。
